#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA50 trend filter and volume spike (>2.0x 20-period average)
# Williams %R identifies overbought/oversold conditions; extreme readings (< -80 for long, > -20 for short) with trend alignment capture reversals.
# 1d EMA50 ensures we trade only with the higher timeframe trend to avoid whipsaws in chop.
# Volume spike confirms institutional participation; discrete sizing (0.25) minimizes fee churn.
# Effective in both bull and bear markets: catches mean reversions in trending markets, avoids false signals in strong trends without volume.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_WilliamsR_1dEMA50_VolumeSpike_v2"
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
    
    # Williams %R on 12h timeframe: (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * (highest_high - close) / (highest_high - lowest_low), 
                          -50.0)  # neutral when range=0
    
    # Calculate 20-period average volume for confirmation (on 12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, period, 20)  # 1d EMA50, Williams %R, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_williams_r = williams_r[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (institutional participation)
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Williams %R conditions
        # Long: Oversold (< -80) with volume confirmation and above 1d EMA50 (uptrend)
        # Short: Overbought (> -20) with volume confirmation and below 1d EMA50 (downtrend)
        williams_long = curr_williams_r < -80.0
        williams_short = curr_williams_r > -20.0
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Williams %R rises above -50 (momentum fading) OR trend turns bearish (price below 1d EMA50)
            if curr_williams_r > -50.0 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (momentum fading) OR trend turns bullish (price above 1d EMA50)
            if curr_williams_r < -50.0 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R oversold AND above 1d EMA50 AND volume confirmation
            if (williams_long and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought AND below 1d EMA50 AND volume confirmation
            elif (williams_short and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals