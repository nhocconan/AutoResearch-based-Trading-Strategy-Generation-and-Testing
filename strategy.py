#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Supertrend for trend direction and price crossing 1d VWAP with volume confirmation
# - Long when price crosses above VWAP with volume spike and 1w Supertrend is bullish
# - Short when price crosses below VWAP with volume spike and 1w Supertrend is bearish
# - Exit when price crosses back below/above VWAP
# - Uses 1d VWAP for mean reversion entry and 1w Supertrend for trend filter
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_VWAP_Supertrend_1w"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Supertrend calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w Supertrend (ATR=10, multiplier=3.0)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([ [high_1w[0] - low_1w[0]], tr ])
    
    # ATR
    atr = np.zeros_like(close_1w)
    atr[9] = np.mean(tr[:10])  # Simple average for first value
    for i in range(10, len(atr)):
        atr[i] = (atr[i-1] * 9 + tr[i]) / 10  # Wilder's smoothing
    
    # Supertrend calculation
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + 3.0 * atr
    lower_band = hl2 - 3.0 * atr
    
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_1w[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align 1w Supertrend direction to 1d timeframe
    supertrend_dir_1d = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate 1d VWAP
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma_20[:10] = np.nan
    vol_ma_20[-10:] = np.nan
    vol_ma_20 = np.where(np.arange(len(volume)) < 10, np.nan,
                         np.where(np.arange(len(volume)) >= len(volume)-10, np.nan, vol_ma_20))
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(supertrend_dir_1d[i]) or np.isnan(vwap[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above VWAP with volume spike and bullish 1w Supertrend
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1] and volume_spike[i] and supertrend_dir_1d[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP with volume spike and bearish 1w Supertrend
            elif close[i] < vwap[i] and close[i-1] >= vwap[i-1] and volume_spike[i] and supertrend_dir_1d[i] == -1:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below VWAP
            if close[i] < vwap[i] and close[i-1] >= vwap[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above VWAP
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals