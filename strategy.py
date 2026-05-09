#142878
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1_S1_TrendFilter_VolumeSpike"
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 30-period EMA on weekly close for trend filter
    close_1w = df_1w['close'].values
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # Calculate weekly CAMARILLA pivot levels from previous weekly bar's OHLC
    prev_1w_high = df_1w['high'].shift(1).values
    prev_1w_low = df_1w['low'].shift(1).values
    prev_1w_close = df_1w['close'].shift(1).values
    
    # Camarilla formula for R1 and S1
    range_1w = prev_1w_high - prev_1w_low
    camarilla_mult = 1.1 / 12  # ~0.0916667
    r1_1w = prev_1w_close + range_1w * camarilla_mult * 1
    s1_1w = prev_1w_close - range_1w * camarilla_mult * 1
    
    # Align Camarilla levels to weekly timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need 30 for weekly EMA and 20 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_30_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_30_1w_aligned[i]
        r1_level = r1_1w_aligned[i]
        s1_level = s1_1w_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Price breaks above R1 with volume AND price > weekly EMA30 (uptrend)
            if close[i] > r1_level and vol > 2.0 * vol_ma_val and close[i] > ema_1w:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S1 with volume AND price < weekly EMA30 (downtrend)
            elif close[i] < s1_level and vol > 2.0 * vol_ma_val and close[i] < ema_1w:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below R1 OR trend reverses (price < weekly EMA30)
            if close[i] < r1_level or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above S1 OR trend reverses (price > weekly EMA30)
            if close[i] > s1_level or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals