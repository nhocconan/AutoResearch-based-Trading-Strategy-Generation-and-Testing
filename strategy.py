#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike
Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter and 1d volume spike filter.
Goes long when price breaks above R1 with 4h uptrend and 1d volume spike.
Goes short when price breaks below S1 with 4h downtrend and 1d volume spike.
Exit when price reverts to Camarilla pivot point (PP). Uses discrete sizing (0.20) to minimize fees.
Target: 15-35 trades/year. Uses HTF for signal direction (4h trend, 1d volume), 1h only for entry timing.
Works in bull via breakouts with trend, in bear via faded rallies at resistance.
"""

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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA34 for trend
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Need daily OHLC - we'll use 1d data shifted by 1 to avoid look-ahead
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to ensure we only use completed daily bars)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align previous day's OHLC to 1h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # PP = (High + Low + Close) / 3
    rang = prev_high_aligned - prev_low_aligned
    r1 = prev_close_aligned + rang * 1.1 / 12
    s1 = prev_close_aligned - rang * 1.1 / 12
    pp = (prev_high_aligned + prev_low_aligned + prev_close_aligned) / 3
    
    # Volume confirmation: 1d volume > 2.0x 20-period average
    vol_spike = volume > (2.0 * vol_ma_20_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(pp[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Long: price breaks above R1, 4h uptrend, 1d volume spike
            long_signal = (close[i] > r1[i]) and (close[i] > ema_34_4h_aligned[i]) and vol_spike[i]
            # Short: price breaks below S1, 4h downtrend, 1d volume spike
            short_signal = (close[i] < s1[i]) and (close[i] < ema_34_4h_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit when price reverts to pivot point (PP)
            exit_signal = close[i] <= pp[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price reverts to pivot point (PP)
            exit_signal = close[i] >= pp[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolSpike"
timeframe = "1h"
leverage = 1.0