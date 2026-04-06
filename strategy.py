#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with volume confirmation and ADX trend filter on 12h timeframe
# Works in bull/bear because breakouts capture strong moves, volume filters weak signals,
# and ADX > 20 ensures we trade in trending conditions only.
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and fee cost.

name = "exp_12976_12h_donchian20_1d_vol_adx_v1"
timezone = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
ADX_PERIOD = 14
ADX_THRESHOLD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smooth = WilderSmooth(tr, period)
    plus_di = 100 * WilderSmooth(plus_dm, period) / tr_smooth
    minus_di = 100 * WilderSmooth(minus_dm, period) / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = WilderSmooth(dx, period)
    
    # Handle division by zero and NaN
    adx = np.where((plus_di + minus_di) == 0, 0, adx)
    adx = np.nan_to_num(adx, nan=0.0)
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Donchian channels
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    upper = np.full_like(high_d, np.nan)
    lower = np.full_like(low_d, np.nan)
    
    for i in range(DONCHIAN_PERIOD - 1, len(high_d)):
        upper[i] = np.max(high_d[i-DONCHIAN_PERIOD+1:i+1])
        lower[i] = np.min(low_d[i-DONCHIAN_PERIOD+1:i+1])
    
    # Calculate daily ADX
    adx_daily = calculate_adx(high_d, low_d, df_daily['close'].values, ADX_PERIOD)
    
    # Align to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_daily, upper)
    lower_aligned = align_htf_to_ltf(prices, df_daily, lower)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx_daily)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Donchian levels not available
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # ADX trend filter
        trend_ok = adx_aligned[i] > ADX_THRESHOLD
        
        # Breakout above upper or breakdown below lower
        breakout_long = volume_ok and trend_ok and close[i] >= upper_aligned[i]
        breakout_short = volume_ok and trend_ok and close[i] <= lower_aligned[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals