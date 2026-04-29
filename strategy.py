#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1d Trend Filter and Volume Spike
# Williams %R identifies overbought/oversold conditions. Extreme readings (< -90 or > -10) 
# often precede reversals. In trending markets (price > 1d EMA34 for longs, < 1d EMA34 for shorts),
# these extremes can signal powerful continuation moves when combined with volume confirmation.
# Uses discrete sizing (0.25) to minimize fee drag. Works in both bull (buy pullbacks in uptrend) 
# and bear (sell rallies in downtrend) markets by aligning with higher timeframe trend.

name = "6h_WilliamsRExtreme_1dEMA34_Trend_v1"
timeframe = "6h"
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
    # Align EMA34 to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 6h data (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    dd = highest_high - lowest_low
    williams_r = np.where(dd != 0, -100 * (highest_high - close) / dd, -50)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 34)  # volume MA, Williams %R, and EMA34 alignment warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34 = ema_34_1d_aligned[i]
        curr_williams = williams_r[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R rises above -50 (overbought) OR price crosses below EMA34
            if curr_williams > -50 or curr_close < curr_ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (oversold) OR price crosses above EMA34
            if curr_williams < -50 or curr_close > curr_ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Williams %R is deeply oversold (< -90) AND volume confirmation 
            # AND price > 1d EMA34 (uptrend filter)
            if curr_williams < -90 and vol_conf and curr_close > curr_ema34:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R is deeply overbought (> -10) 
            # AND volume confirmation AND price < 1d EMA34 (downtrend filter)
            elif curr_williams > -10 and vol_conf and curr_close < curr_ema34:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals