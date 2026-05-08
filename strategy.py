#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Alligator_Teeth_Slope_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6h Williams Alligator (13,8,5 smoothed by 8,5,3)
    jaw_period, teeth_period, lips_period = 13, 8, 5
    jaw_shift, teeth_shift, lips_shift = 8, 5, 3
    
    # Jaw: SMMA(13, 8)
    jaw = pd.Series(close).ewm(alpha=1/jaw_period, adjust=False).mean().values
    jaw = pd.Series(jaw).ewm(alpha=1/jaw_shift, adjust=False).mean().values
    
    # Teeth: SMMA(8, 5) - this is our primary signal line
    teeth = pd.Series(close).ewm(alpha=1/teeth_period, adjust=False).mean().values
    teeth = pd.Series(teeth).ewm(alpha=1/teeth_shift, adjust=False).mean().values
    
    # Lips: SMMA(5, 3) - for confirmation
    lips = pd.Series(close).ewm(alpha=1/lips_period, adjust=False).mean().values
    lips = pd.Series(lips).ewm(alpha=1/lips_shift, adjust=False).mean().values
    
    # Teeth slope (3-period change) - indicates turning strength
    teeth_slope = teeth - np.roll(teeth, 3)
    teeth_slope[:3] = 0
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, jaw_shift + teeth_shift + lips_shift + 3)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(teeth_slope[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: teeth > lips AND teeth slope turning up AND uptrend (price > 1d EMA34) + volume spike
            long_cond = (teeth[i] > lips[i]) and \
                        (teeth_slope[i] > 0) and \
                        (close[i] > ema_34_1d_aligned[i]) and \
                        volume_spike[i]
            # Short: teeth < lips AND teeth slope turning down AND downtrend (price < 1d EMA34) + volume spike
            short_cond = (teeth[i] < lips[i]) and \
                         (teeth_slope[i] < 0) and \
                         (close[i] < ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: teeth crosses below lips (Alligator sleeping) OR price breaks 1d EMA
            if (teeth[i] < lips[i]) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: teeth crosses above lips OR price breaks 1d EMA
            if (teeth[i] > lips[i]) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals