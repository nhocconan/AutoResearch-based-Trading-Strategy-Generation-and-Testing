#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3)
# Long: Lips > Teeth > Jaw AND price > 1d EMA34 AND volume > 1.5x 20-bar avg
# Short: Lips < Teeth < Jaw AND price < 1d EMA34 AND volume > 1.5x 20-bar avg
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
# Alligator identifies trend initiation/continuation; EMA34 filters for higher timeframe trend; volume confirms momentum

name = "12h_Williams_Alligator_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (SMMA with smoothing)
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Jaw: 13-period SMMA smoothed 8
    jaw_raw = smma(close, 13)
    jaw = smma(jaw_raw, 8)
    # Teeth: 8-period SMMA smoothed 5
    teeth_raw = smma(close, 8)
    teeth = smma(teeth_raw, 5)
    # Lips: 5-period SMMA smoothed 3
    lips_raw = smma(close, 5)
    lips = smma(lips_raw, 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Alligator components
    
    for i in range(start_idx, n):
        # Skip if Alligator values not ready
        if np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Alligator alignment
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Alligator loses alignment (Lips < Teeth OR Teeth < Jaw) OR price < 1d EMA34
            if not (lips_above_teeth and teeth_above_jaw) or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator loses alignment (Lips > Teeth OR Teeth > Jaw) OR price > 1d EMA34
            if not (lips_below_teeth and teeth_below_jaw) or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Lips > Teeth > Jaw AND price > 1d EMA34 AND volume confirmation
            if (lips_above_teeth and teeth_above_jaw and
                curr_close > curr_ema_1d and
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaw AND price < 1d EMA34 AND volume confirmation
            elif (lips_below_teeth and teeth_below_jaw and
                  curr_close < curr_ema_1d and
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals