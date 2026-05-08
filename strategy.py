#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and daily ATR filter.
# Uses 20-period Donchian channels on 4h for breakout signals.
# Volume confirmation: current volume > 1.5x 20-period volume EMA.
# ATR filter: only trade when 4h ATR(14) > daily ATR(14) * 0.5 (ensures sufficient volatility).
# Trend filter: 4h EMA(50) direction.
# Long when price breaks above upper Donchian with volume confirmation and uptrend.
# Short when price breaks below lower Donchian with volume confirmation and downtrend.
# Exit when price crosses opposite Donchian band or trend changes.
# Designed for low trade frequency (20-40/year) to minimize fee drag while capturing trends.

name = "4h_Donchian_Breakout_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) on 4h
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate ATR(14) on daily
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align daily ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: 4h volume > 1.5x 20-period volume EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    # Trend filter: 4h EMA(50)
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ATR filter: only trade when 4h ATR > daily ATR * 0.5
        atr_filter = atr_4h[i] > (atr_1d_aligned[i] * 0.5)
        
        if position == 0:
            # Long entry: price breaks above upper Donchian with volume confirmation and uptrend
            if close[i] > donchian_high[i] and vol_confirm[i] and atr_filter:
                if close[i] > ema_50[i]:  # Uptrend filter
                    signals[i] = 0.25
                    position = 1
            # Short entry: price breaks below lower Donchian with volume confirmation and downtrend
            elif close[i] < donchian_low[i] and vol_confirm[i] and atr_filter:
                if close[i] < ema_50[i]:  # Downtrend filter
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below lower Donchian or trend turns down
            if close[i] < donchian_low[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above upper Donchian or trend turns up
            if close[i] > donchian_high[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals