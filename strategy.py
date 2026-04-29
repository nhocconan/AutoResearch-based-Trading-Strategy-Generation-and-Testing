#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 Breakout + 1d EMA34 Trend + Volume Spike
# Long when price breaks above Camarilla R1 level AND price > 1d EMA34 AND volume > 1.8x 20-bar avg
# Short when price breaks below Camarilla S1 level AND price < 1d EMA34 AND volume > 1.8x 20-bar avg
# Exit when price reverts to Camarilla Pivot level (mean reversion)
# Uses discrete position sizing (0.25) to balance return and fee drag. Target: 20-50 trades/year on 4h timeframe.
# Camarilla levels provide precise intraday support/resistance, 1d EMA34 filters counter-trend moves,
# volume confirmation ensures breakout strength.

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Get 1d data for Camarilla pivot calculation and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d data for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_high = df_1d['high'].shift(1).values  # Previous day high
    prev_low = df_1d['low'].shift(1).values    # Previous day low
    prev_close = df_1d['close'].shift(1).values # Previous day close
    
    # Handle first bar where shift creates NaN
    prev_high[0] = df_1d['high'].iloc[0]
    prev_low[0] = df_1d['low'].iloc[0]
    prev_close[0] = df_1d['close'].iloc[0]
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_R1 = prev_close + camarilla_range * 1.1 / 12.0
    camarilla_S1 = prev_close - camarilla_range * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume MA and EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34 = ema_34_1d_aligned[i]
        curr_R1 = camarilla_R1_aligned[i]
        curr_S1 = camarilla_S1_aligned[i]
        curr_pivot = camarilla_pivot_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price reverts to Camarilla Pivot level (mean reversion)
            if curr_close <= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to Camarilla Pivot level (mean reversion)
            if curr_close >= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Camarilla R1 AND price > 1d EMA34 AND volume confirmation
            if curr_close > curr_R1 and curr_close > curr_ema34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S1 AND price < 1d EMA34 AND volume confirmation
            elif curr_close < curr_S1 and curr_close < curr_ema34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals