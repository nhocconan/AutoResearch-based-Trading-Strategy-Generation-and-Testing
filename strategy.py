#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WilliamsAlligator_R1S1_Breakout_1wTrend"
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
    
    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Get daily data for Williams Alligator and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Williams Alligator: SMA(13,8), SMA(8,5), SMA(5,3)
    close_1d_series = pd.Series(df_1d['close'])
    jaw = close_1d_series.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = close_1d_series.rolling(window=8, min_periods=8).mean().shift(5).values
    lips = close_1d_series.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Williams %R: (highest_high - close) / (highest_high - lowest_low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - df_1d['close'].values) / (highest_high - lowest_low + 1e-10)) * -100
    
    # Williams %R levels: -20 (overbought), -80 (oversold)
    # Williams %R signals: above -20 = overbought, below -80 = oversold
    
    # 1d Williams Alligator alignment
    jaw_1d = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_1d = align_htf_to_ltf(prices, df_1d, teeth)
    lips_1d = align_htf_to_ltf(prices, df_1d, lips)
    williams_r_1d = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Weekly trend: price above/below 34-period EMA
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    camarilla_high = close_1d + 1.1 * range_1d / 12  # R1 level
    camarilla_low = close_1d - 1.1 * range_1d / 12   # S1 level
    camarilla_high_1d = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_1d = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Daily volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_1d[i]) or np.isnan(teeth_1d[i]) or np.isnan(lips_1d[i]) or 
            np.isnan(williams_r_1d[i]) or np.isnan(ema34_1w_aligned[i]) or
            np.isnan(camarilla_high_1d[i]) or np.isnan(camarilla_low_1d[i]) or
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator conditions
        jaw_val = jaw_1d[i]
        teeth_val = teeth_1d[i]
        lips_val = lips_1d[i]
        williams_r_val = williams_r_1d[i]
        ema34_1w_val = ema34_1w_aligned[i]
        resistance = camarilla_high_1d[i]
        support = camarilla_low_1d[i]
        vol_avg = vol_avg_1d_aligned[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        # Williams Alligator: bullish when lips > teeth > jaw, bearish when lips < teeth < jaw
        bullish_alligator = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alligator = lips_val < teeth_val and teeth_val < jaw_val
        
        # Williams %R: oversold < -80, overbought > -20
        oversold = williams_r_val < -80
        overbought = williams_r_val > -20
        
        if position == 0:
            # Long: bullish Alligator + oversold Williams %R + price above Camarilla R1 + weekly uptrend + volume
            if (bullish_alligator and oversold and close[i] > resistance and 
                close[i] > ema34_1w_val and vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator + overbought Williams %R + price below Camarilla S1 + weekly downtrend + volume
            elif (bearish_alligator and overbought and close[i] < support and 
                  close[i] < ema34_1w_val and vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish Alligator crossover OR price below Camarilla S1
            if bearish_alligator or close[i] < support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish Alligator crossover OR price above Camarilla R1
            if bullish_alligator or close[i] > resistance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals