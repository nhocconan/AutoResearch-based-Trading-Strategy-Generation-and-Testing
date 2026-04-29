#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 Breakout + 4h EMA50 Trend + Volume Spike + Session Filter (08-20 UTC)
# Long when price breaks above Camarilla R1 level AND price > 4h EMA50 AND volume > 2.0x 20-bar avg AND session filter
# Short when price breaks below Camarilla S1 level AND price < 4h EMA50 AND volume > 2.0x 20-bar avg AND session filter
# Exit when price reverts to Camarilla Pivot level (mean reversion)
# Uses discrete position sizing (0.20) to reduce fee drag. Target: 15-37 trades/year on 1h timeframe.
# Camarilla levels provide precise intraday support/resistance, 4h EMA50 filters counter-trend moves,
# volume confirmation ensures breakout strength, session filter reduces noise trades.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 1d data for Camarilla pivot calculation (using daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA(50) on 4h data
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
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
    
    # Align Camarilla levels to 1h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i] and session_filter[i]
        curr_ema50 = ema_50_4h_aligned[i]
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
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price reverts to Camarilla Pivot level (mean reversion)
            if curr_close >= curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long when price breaks above Camarilla R1 AND price > 4h EMA50 AND volume confirmation AND session
            if curr_close > curr_R1 and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below Camarilla S1 AND price < 4h EMA50 AND volume confirmation AND session
            elif curr_close < curr_S1 and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals