#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA34 Trend + Volume Spike
# Williams %R below -80 = oversold, above -20 = overbought
# Long when Williams %R crosses above -80 from below AND price > 1d EMA34 AND volume > 1.8x 20-bar avg
# Short when Williams %R crosses below -20 from above AND price < 1d EMA34 AND volume > 1.8x 20-bar avg
# Exit when Williams %R returns to neutral zone (-50) or opposite extreme
# Uses discrete position sizing (0.25) to minimize fee drag.
# Williams %R captures momentum extremes, volume confirmation ensures follow-through,
# 1d EMA34 filters counter-trend moves. Works in ranging markets (mean reversion from extremes)
# and trending markets (pullbacks to EMA in direction of trend).

name = "6h_WilliamsRExtreme_1dEMA34_VolumeSpike_v1"
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
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    williams_r = np.where(rr != 0, -100 * (highest_high - close) / rr, -50)
    
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
            np.isnan(volume_ma_20[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_williams = williams_r[i]
        curr_ema34 = ema_34_1d_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Calculate previous Williams %R for crossover detection
        prev_williams = williams_r[i-1] if i > 0 else -50
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R returns to neutral zone (-50) or reaches overbought (-20)
            if curr_williams >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral zone (-50) or reaches oversold (-80)
            if curr_williams <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Williams %R crosses above -80 from below (ending oversold)
            # AND price > 1d EMA34 AND volume confirmation
            if prev_williams < -80 and curr_williams >= -80 and curr_close > curr_ema34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R crosses below -20 from above (ending overbought)
            # AND price < 1d EMA34 AND volume confirmation
            elif prev_williams > -20 and curr_williams <= -20 and curr_close < curr_ema34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals