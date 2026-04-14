#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot (from 1-day) with volume confirmation and ADX trend filter.
# Camarilla levels derived from prior 1-day range (H-L-C) act as intraday support/resistance.
# Long at L3 with bullish trend (ADX>20) and volume >1.5x average; short at H3 with bearish trend.
# Exit when price reaches opposite H3/L3 level or closes back inside the Camarilla range.
# Designed for low-frequency, high-conviction trades (~15-25/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    ph = df_1d['high'].shift(1).values  # prior day high
    pl = df_1d['low'].shift(1).values   # prior day low
    pc = df_1d['close'].shift(1).values # prior day close
    range_ = ph - pl
    
    # Camarilla levels
    H3 = pc + (range_ * 1.1 / 4)
    L3 = pc - (range_ * 1.1 / 4)
    H4 = pc + (range_ * 1.1 / 2)
    L4 = pc - (range_ * 1.1 / 2)
    
    # Align to 12h timeframe
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # ADX(14) for trend strength on 12h
    adx_len = 14
    if len(prices) < adx_len + 1:
        return np.zeros(n)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/adx_len, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/adx_len, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/adx_len, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/adx_len, adjust=False).mean().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position
    
    start = max(adx_len, 20)  # ensure ADX and volume MA are ready
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H3_12h[i]) or np.isnan(L3_12h[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx[i] > 20
        
        if position == 0:
            # Enter long at L3 with bullish bias
            if (close[i] <= L3_12h[i] and 
                trending and 
                volume_confirmed and
                plus_di[i] > minus_di[i]):  # bullish bias
                position = 1
                signals[i] = position_size
            # Enter short at H3 with bearish bias
            elif (close[i] >= H3_12h[i] and 
                  trending and 
                  volume_confirmed and
                  minus_di[i] > plus_di[i]):  # bearish bias
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H4 or closes back inside Camarilla (L3-H3)
            if close[i] >= H4_12h[i] or (close[i] >= L3_12h[i] and close[i] <= H3_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L4 or closes back inside Camarilla (L3-H3)
            if close[i] <= L4_12h[i] or (close[i] >= L3_12h[i] and close[i] <= H3_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_ADX_Volume_v1"
timeframe = "12h"
leverage = 1.0