#!/usr/bin/env python3
name = "1h_4h_1d_Donchian_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h Donchian channels (20 periods)
    donch_high_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA trend (34 periods)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1h volume spike detection (24-period average = 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align HTF indicators to 1h timeframe
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Session filter: 8-20 UTC (pre-market to post-US close)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 24, 34)  # Wait for all indicators
    
    for i in range(start_idx, n):
        # Skip if any indicator is NaN
        if (np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        if position == 0:
            # Long: price breaks above 4h Donchian high with volume and 1d uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > donch_high_4h_aligned[i] and vol_condition and uptrend and in_session:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low with volume and 1d downtrend
            elif close[i] < donch_low_4h_aligned[i] and vol_condition and not uptrend and in_session:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price returns to 4h Donchian low or volume drops
            if close[i] < donch_low_4h_aligned[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price returns to 4h Donchian high or volume drops
            if close[i] > donch_high_4h_aligned[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Donchian breakout with 4h structure and 1d trend filter
# - 4h Donchian channels (20-period) provide structural support/resistance
# - Breakout above 4h Donchian high with volume spike in 1d uptrend = long
# - Breakdown below 4h Donchian low with volume spike in 1d downtrend = short
# - Volume confirmation (2x average) filters false breakouts
# - Session filter (8-20 UTC) avoids low-liquidity Asian session
# - Position size 0.20 targets 15-35 trades/year to avoid fee drag
# - Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)