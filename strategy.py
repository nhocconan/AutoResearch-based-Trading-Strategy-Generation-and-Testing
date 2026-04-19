#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_Volume_Trend_v2"
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
    
    # Get 1w data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    hl_range = high_1w - low_1w
    camarilla_r1 = close_1w + 1.1 * hl_range / 12
    camarilla_s1 = close_1w - 1.1 * hl_range / 12
    
    # Align Camarilla levels to 1d timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Calculate 1d EMA34 for trend filter
    close_s = pd.Series(close)
    ema34 = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(ema34[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema = ema34[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend filter: bullish when price > EMA34
        bullish_trend = price > ema
        bearish_trend = price < ema
        
        if position == 0:
            # Long: price breaks above R1 in bullish trend with volume
            if price > r1 and bullish_trend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in bearish trend with volume
            elif price < s1 and bearish_trend and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below S1 or trend changes
            if price < s1 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above R1 or trend changes
            if price > r1 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals