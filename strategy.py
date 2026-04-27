#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Calculate 14-period Choppiness Index
    atr14 = np.zeros(len(df_1d))
    for i in range(14, len(df_1d)):
        tr = max(df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                 abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                 abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1]))
        atr14[i] = (atr14[i-1] * 13 + tr) / 14
    atr14[:14] = np.nan
    
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    chop = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if not np.isnan(atr14[i]) and not np.isnan(highest_high[i]) and not np.isnan(lowest_low[i]):
            sum_atr = atr14[i] * 14
            if highest_high[i] - lowest_low[i] > 0:
                chop[i] = 100 * np.log10(sum_atr / (highest_high[i] - lowest_low[i])) / np.log10(14)
    
    chop_ma = pd.Series(chop).rolling(window=5, min_periods=5).mean().values
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_ma)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need chop data, 4h EMA, and volume data
    start_idx = max(14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        ema_trend = ema50_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5 * 20-period average (volume surge)
        vol_filter = vol_current > (vol_ma_val * 1.5)
        
        # Regime filter: Chop > 61.8 = ranging market (mean reversion)
        ranging = chop_val > 61.8
        
        if position == 0:
            # Long: price touches lower Bollinger Band in ranging market with volume surge
            # Calculate Bollinger Bands (20, 2) on 4h close
            if i >= 20:
                bb_middle = np.mean(close[i-19:i+1])
                bb_std = np.std(close[i-19:i+1])
                bb_lower = bb_middle - 2 * bb_std
                
                if close[i] <= bb_lower and ranging and vol_filter:
                    signals[i] = size
                    position = 1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches middle Bollinger Band or trend turns down
            if i >= 20:
                bb_middle = np.mean(close[i-19:i+1])
                if close[i] >= bb_middle or close[i] < ema_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            else:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Exit short: price touches middle Bollinger Band or trend turns up
            if i >= 20:
                bb_middle = np.mean(close[i-19:i+1])
                if close[i] >= bb_middle or close[i] > ema_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            else:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Choppiness_BB_MeanReversion_VolumeSurge"
timeframe = "4h"
leverage = 1.0