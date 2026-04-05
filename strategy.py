#!/usr/bin/env python3
"""
Experiment #8814: 1h Donchian breakout + 4h/1d trend filter + volume confirmation + ATR stoploss.
Hypothesis: Use 4h and 1d timeframes for trend direction, 1h for entry timing to balance signal quality and trade frequency.
Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag while maintaining statistical validity.
Session filter: 08-20 UTC to avoid low-liquidity hours.
Position size: 0.20 (20% of capital).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8814_1h_donchian20_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_4H_PERIOD = 50
TREND_1D_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
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

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h and 1d EMA for trend filter
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=TREND_4H_PERIOD, adjust=False, min_periods=TREND_4H_PERIOD).mean().values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_1D_PERIOD, adjust=False, min_periods=TREND_1D_PERIOD).mean().values
    
    # Price relative to EMAs: above = bullish bias, below = bearish bias
    trend_4h = np.where(close_4h > ema_4h, 1, 
                   np.where(close_4h < ema_4h, -1, 0))
    trend_1d = np.where(close_1d > ema_1d, 1, 
                    np.where(close_1d < ema_1d, -1, 0))
    
    # Align to 1h timeframe
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_4H_PERIOD, TREND_1D_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine market bias from 4h and 1d EMAs (both must agree)
        bull_bias = (trend_4h_aligned[i] == 1) and (trend_1d_aligned[i] == 1)
        bear_bias = (trend_4h_aligned[i] == -1) and (trend_1d_aligned[i] == -1)
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_bias and long_breakout and volume_confirmed
        short_entry = bear_bias and short_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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