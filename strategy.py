#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot reversal with daily volume confirmation and ADX trend filter.
# Captures mean-reversion at institutional pivot levels during ranging markets while avoiding strong trends.
# Works in bull markets (reversions from overbought pivots) and bear markets (reversions from oversold pivots).
# Target: 75-200 total trades over 4 years (19-50/year).

name = "exp_13563_4h_camarilla1d_adx_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    for i in range(1, len(high)):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        elif down > up and down > 0:
            minus_dm[i] = down
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

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
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day)
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1 * range_1d / 2
    camarilla_l4 = close_1d - 1.1 * range_1d / 2
    camarilla_h3 = close_1d + 1.1 * range_1d / 4
    camarilla_l3 = close_1d - 1.1 * range_1d / 4
    
    camarilla_h4_prev = np.roll(camarilla_h4, 1)
    camarilla_l4_prev = np.roll(camarilla_l4, 1)
    camarilla_h3_prev = np.roll(camarilla_h3, 1)
    camarilla_l3_prev = np.roll(camarilla_l3, 1)
    camarilla_h4_prev[0] = np.nan
    camarilla_l4_prev[0] = np.nan
    camarilla_h3_prev[0] = np.nan
    camarilla_l3_prev[0] = np.nan
    
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_prev)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_prev)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_prev)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_prev)
    
    # Calculate daily ADX for trend filter
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4-hour indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if data not available
        if np.isnan(adx_1d_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or \
           np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or \
           np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
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
        
        # Trend filter: only trade when ADX < threshold (ranging market)
        ranging = adx_1d_aligned[i] < ADX_THRESHOLD
        
        # Mean reversion signals at Camarilla levels
        long_signal = volume_ok and ranging and (low[i] <= camarilla_l3_aligned[i]) and (close[i] > camarilla_l3_aligned[i])
        short_signal = volume_ok and ranging and (high[i] >= camarilla_h3_aligned[i]) and (close[i] < camarilla_h3_aligned[i])
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
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