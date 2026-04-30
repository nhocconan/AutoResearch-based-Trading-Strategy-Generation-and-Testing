#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume spike confirmation.
# Uses 1d HTF for EMA34 trend and Elder Ray calculation (EMA13 of close).
# Long when Bull Power > 0 AND price > 1d EMA34 AND volume > 2.0x 20-bar average.
# Short when Bear Power < 0 AND price < 1d EMA34 AND volume > 2.0x 20-bar average.
# Exit when power crosses zero or trend filter fails.
# Discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Works in bull/bear via 1d EMA34 trend filter and volume confirmation to avoid false signals.

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray components on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = df_1d['low'].values - ema_13_1d
    
    # Align 1d indicators to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA34, EMA13 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0, uptrend (price > 1d EMA34), volume confirmation
            if (bull_power_1d_aligned[i] > 0 and 
                curr_close > ema_34_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, downtrend (price < 1d EMA34), volume confirmation
            elif (bear_power_1d_aligned[i] < 0 and 
                  curr_close < ema_34_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: Bull Power <= 0 OR trend fails OR volume confirmation lost
            if (bull_power_1d_aligned[i] <= 0 or 
                curr_close <= ema_34_1d_aligned[i] or 
                not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Bear Power >= 0 OR trend fails OR volume confirmation lost
            if (bear_power_1d_aligned[i] >= 0 or 
                curr_close >= ema_34_1d_aligned[i] or 
                not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals