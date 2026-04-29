#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume confirmation (>2.0x 20-period average)
# Williams Alligator identifies trend absence (all lines intertwined) vs presence (lines aligned).
# In trending markets: Jaw (13-period) > Teeth (8-period) > Lips (5-period) for uptrend; reverse for downtrend.
# Combined with 1d EMA50 for higher timeframe trend alignment and volume confirmation for institutional participation.
# Discrete sizing (0.25) minimizes fee churn. Works in both bull/bear markets by capturing strong trends.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_Williams_Alligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator (12h timeframe)
    # Jaw: 13-period SMMA, smoothed by 8 periods
    # Teeth: 8-period SMMA, smoothed by 5 periods  
    # Lips: 5-period SMMA, smoothed by 3 periods
    close_s = pd.Series(close)
    
    # SMMA (Smoothed Moving Average) implementation
    def smma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        sma = values.rolling(window=period, min_periods=period).mean()
        result[period-1] = sma.iloc[period-1]
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values.iloc[i]) / period
        return result
    
    jaw = smma(close_s, 13)
    # Smoothed by 8 periods
    jaw_smooth = pd.Series(jaw).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    teeth = smma(close_s, 8)
    # Smoothed by 5 periods
    teeth_smooth = pd.Series(teeth).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    lips = smma(close_s, 5)
    # Smoothed by 3 periods
    lips_smooth = pd.Series(lips).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Calculate 20-period average volume for confirmation (on 12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13+8, 8+5, 5+3, 20)  # 1d EMA50, Alligator components, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_smooth[i]) or 
            np.isnan(teeth_smooth[i]) or np.isnan(lips_smooth[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_jaw = jaw_smooth[i]
        curr_teeth = teeth_smooth[i]
        curr_lips = lips_smooth[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Alligator lines reverse (teeth crosses below lips) OR price below 1d EMA50
            if curr_teeth < curr_lips or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines reverse (teeth crosses above lips) OR price above 1d EMA50
            if curr_teeth > curr_lips or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Lips > Teeth > Jaw (aligned up) + above 1d EMA50 + volume confirmation
            if (curr_lips > curr_teeth and 
                curr_teeth > curr_jaw and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaw (aligned down) + below 1d EMA50 + volume confirmation
            elif (curr_lips < curr_teeth and 
                  curr_teeth < curr_jaw and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals