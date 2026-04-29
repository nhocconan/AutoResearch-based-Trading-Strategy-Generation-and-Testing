#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze Breakout with 1d EMA34 trend filter and volume confirmation (>1.8x 20-period average)
# Bollinger Band squeeze (low volatility) precedes explosive moves; breakout direction filtered by 1d EMA34
# Volume confirmation ensures institutional participation; discrete sizing (0.25) minimizes fee churn
# Works in both bull/bear markets: squeeze captures volatility contraction/expansion cycles universal across regimes
# Target: 80-180 total trades over 4 years (20-45/year) on 4h timeframe

name = "4h_BB_Squeeze_Breakout_1dEMA34_VolumeConfirm_v2"
timeframe = "4h"
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
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Bollinger Bands (20, 2.0) on 4h timeframe
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # normalized width
    
    # Bollinger Band Squeeze: width below 20-period average width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Calculate 20-period average volume for confirmation (on 4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 20)  # 1d EMA34, BB middle, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_upper = bb_upper[i]
        curr_lower = bb_lower[i]
        curr_squeeze = squeeze[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = curr_volume > 1.8 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below BB middle OR squeeze re-activates (volatility contraction)
            if curr_close < bb_middle[i] or curr_squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above BB middle OR squeeze re-activates
            if curr_close > bb_middle[i] or curr_squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: breakout above BB upper + above 1d EMA34 + volume confirmation + squeeze release
            if (curr_close > curr_upper and 
                curr_close > curr_ema_1d and 
                vol_confirm and 
                not curr_squeeze):  # squeeze released (expanding volatility)
                signals[i] = 0.25
                position = 1
            # Short entry: breakout below BB lower + below 1d EMA34 + volume confirmation + squeeze release
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_1d and 
                  vol_confirm and 
                  not curr_squeeze):  # squeeze released
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals