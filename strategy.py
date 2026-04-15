#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (13,8,5 SMAs) with 1d volume confirmation and 1w trend filter
# Williams Alligator identifies trending vs ranging markets: jaws (13), teeth (8), lips (5)
# In trending markets: lips > teeth > jaws (uptrend) or lips < teeth < jaws (downtrend)
# Entry: Alligator aligned + price outside lips + volume spike + 1w EMA trend filter
# Exit: Price crosses lips or Alligator convergence (jaws-teeth-lips intertwine)
# Designed for low trade frequency (target 15-30/year) with clear trend following logic
# Works in both bull (trend continuation) and bear (trend continuation) markets
# Uses volume spike to confirm breakout strength and avoid false signals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Williams Alligator calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Williams Alligator: 3 smoothed SMAs
    # Jaws: 13-period SMA, smoothed by 8 periods
    sma13_6h = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().values
    jaws_6h = pd.Series(sma13_6h).rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMA, smoothed by 5 periods
    sma8_6h = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().values
    teeth_6h = pd.Series(sma8_6h).rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMA, smoothed by 3 periods
    sma5_6h = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().values
    lips_6h = pd.Series(sma5_6h).rolling(window=3, min_periods=3).mean().values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR for volatility and position sizing (14-period on 6h)
    tr1 = np.maximum(high_6h[1:], low_6h[:-1]) - np.minimum(high_6h[1:], low_6h[:-1])
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to 6h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_6h, jaws_6h)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth_6h)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips_6h)
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
        
        # Alligator alignment checks
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_above_jaws = teeth_aligned[i] > jaws_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_below_jaws = teeth_aligned[i] < jaws_aligned[i]
        
        # Alligator convergence ( jaws, teeth, lips intertwined = ranging )
        alligator_convergence = (
            abs(jaws_aligned[i] - teeth_aligned[i]) < 0.001 * close[i] and
            abs(teeth_aligned[i] - lips_aligned[i]) < 0.001 * close[i] and
            abs(jaws_aligned[i] - lips_aligned[i]) < 0.001 * close[i]
        )
        
        # Long entry: Alligator aligned up + price above lips + volume spike + uptrend
        if (lips_above_teeth and teeth_above_jaws and 
            close[i] > lips_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            not alligator_convergence and
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: Alligator aligned down + price below lips + volume spike + downtrend
        elif (lips_below_teeth and teeth_below_jaws and 
              close[i] < lips_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              not alligator_convergence and
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: Price crosses lips OR Alligator convergence (market ranging)
        elif position == 1 and (close[i] < lips_aligned[i] or alligator_convergence):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > lips_aligned[i] or alligator_convergence):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_1dVolume_1wEMA_Trend"
timeframe = "6h"
leverage = 1.0