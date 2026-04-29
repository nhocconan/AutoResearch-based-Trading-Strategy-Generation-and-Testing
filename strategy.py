#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme Reversion with 1d EMA34 trend filter and volume spike
# Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when %R < -80 (oversold) AND price > 1d EMA34 AND volume > 2x 20-bar avg
# Short when %R > -20 (overbought) AND price < 1d EMA34 AND volume > 2x 20-bar avg
# Exit when %R crosses back above -50 (for longs) or below -50 (for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 75-200 total trades over 4 years (19-50/year) on 4h.
# Williams %R identifies exhaustion points; reversal after extreme readings works in both bull and bear markets.
# 1d EMA34 filters counter-trend moves, volume confirmation ensures participation.

name = "4h_WilliamsRExtreme_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    williams_r = np.where(hh_ll != 0, (highest_high - close) / hh_ll * -100, -50)
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 34)  # Williams %R warmup and EMA34 alignment
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_34 = ema_34_1d_aligned[i]
        curr_wr = williams_r[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R crosses back above -50 (momentum fading)
            if curr_wr >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses back below -50 (momentum fading)
            if curr_wr <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND price > 1d EMA34 AND volume confirmation
            if curr_wr < -80 and close[i] > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) AND price < 1d EMA34 AND volume confirmation
            elif curr_wr > -20 and close[i] < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals