#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop
Hypothesis: On 1d timeframe, enter long when price breaks above 20-day Donchian high with 1-week uptrend (close > EMA50) and volume spike (>2.0x 20-day average). Enter short when price breaks below 20-day Donchian low with 1-week downtrend and volume spike. Exit via ATR-based trailing stop (3x ATR) or opposite breakout. Uses discrete position size 0.25 to limit drawdown. Designed for 7-25 trades/year on 1d by requiring weekly alignment and volume confirmation, reducing fee drag while capturing structured moves in bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and 1w for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels (based on previous 20 completed 1d bars)
    high_1d_series = pd.Series(df_1d['high'].values)
    low_1d_series = pd.Series(df_1d['low'].values)
    donchian_high = high_1d_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_1d_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (no additional delay needed as they're based on completed 1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1-week EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # ATR for trailing stop (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_series = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup: need Donchian warmup, EMA warmup, volume MA warmup, ATR warmup
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr_series[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend alignment
        trend_1w_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + 1w uptrend + volume spike
            long_signal = (close[i] > donchian_high_aligned[i]) and trend_1w_uptrend and volume_spike[i]
            
            # Short: price breaks below Donchian low + 1w downtrend + volume spike
            short_signal = (close[i] < donchian_low_aligned[i]) and trend_1w_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # ATR trailing stop: exit if price drops 3*ATR from highest high
            atr_stop = highest_high_since_entry - 3.0 * atr_series[i]
            # Exit: ATR stop hit OR price breaks below Donchian low OR 1w trend turns down
            if (close[i] <= atr_stop or close[i] < donchian_low_aligned[i] or not trend_1w_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # ATR trailing stop: exit if price rises 3*ATR from lowest low
            atr_stop = lowest_low_since_entry + 3.0 * atr_series[i]
            # Exit: ATR stop hit OR price breaks above Donchian high OR 1w trend turns up
            if (close[i] >= atr_stop or close[i] > donchian_high_aligned[i] or not trend_1w_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0