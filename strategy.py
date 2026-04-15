#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d volume filter and 1w trend alignment
# Uses Alligator jaws/teeth/lips for trend detection, volume spikes for confirmation,
# and weekly EMA for trend alignment. Works in trending markets (both bull/bear)
# and avoids whipsaws in ranging conditions. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Williams Alligator components (13,8,5 smoothed with 8,5,3)
    # Jaws: 13-period SMMA smoothed by 8
    jaws_raw = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().values
    jaws = pd.Series(jaws_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMMA smoothed by 5
    teeth_raw = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMMA smoothed by 3
    lips_raw = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR for volatility and stoploss (14-period on 6h)
    tr1 = np.maximum(high_6h[1:], low_6h[:-1]) - np.minimum(high_6h[1:], low_6h[:-1])
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to 6h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_6h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_aligned[i])):
            continue
        
        # Volatility-adjusted position size (inverse vol)
        vol_factor = np.clip(0.5 * atr_aligned[i] / (close[i] + 1e-10), 0.5, 2.0)
        position_size = base_size / vol_factor
        position_size = np.clip(position_size, 0.15, 0.35)
        
        # Alligator alignment: jaws > teeth > lips = uptrend, reverse for downtrend
        jaws_teeth_lips_up = jaws_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        jaws_teeth_lips_down = jaws_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        
        # Long entry: Alligator aligned up + price above lips + volume spike + above weekly EMA
        if (jaws_teeth_lips_up and 
            close[i] > lips_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: Alligator aligned down + price below lips + volume spike + below weekly EMA
        elif (jaws_teeth_lips_down and 
              close[i] < lips_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: Alligator reversal or price crosses teeth
        elif position == 1 and (jaws_aligned[i] < teeth_aligned[i] or 
                                close[i] < teeth_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (jaws_aligned[i] > teeth_aligned[i] or 
                                 close[i] > teeth_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_1dVolume_1wEMA_Trend"
timeframe = "6h"
leverage = 1.0