#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Mean Reversion with 1d EMA34 trend filter and volume confirmation
# Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long: Williams %R < -80 (oversold) AND price > 1d EMA34 (bullish trend) AND volume spike (>1.8x 24-bar average)
# Short: Williams %R > -20 (overbought) AND price < 1d EMA34 (bearish trend) AND volume spike
# Uses mean reversion in ranging markets + trend filter to avoid counter-trend in strong moves
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_WilliamsR_MeanRev_1dEMA34_VolumeConfirm_v1"
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
    
    # Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100,
        -50  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # max(14, 34) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams = williams_r[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.8x 24-period average
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        if np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 1.8 * vol_ma_24[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Williams %R >= -50 (exiting oversold) OR price below 1d EMA34 (trend change)
            if curr_williams >= -50 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R <= -50 (exiting overbought) OR price above 1d EMA34 (trend change)
            if curr_williams <= -50 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) AND price above 1d EMA34 AND volume spike
            if (curr_williams < -80 and 
                curr_close > curr_ema_1d and
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought) AND price below 1d EMA34 AND volume spike
            elif (curr_williams > -20 and 
                  curr_close < curr_ema_1d and
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals