#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) for trend bias and 1d Camarilla pivot (R1/S1) for mean reversion entries
# 4h Donchian(20) determines trend: long bias when price > upper channel, short bias when price < lower channel
# 1d Camarilla R1/S1 levels provide high-probability mean reversion entries in ranging markets
# Volume confirmation (1.5x 20-period average) filters low-quality breakouts
# Session filter (08-20 UTC) avoids Asian session noise
# Discrete position sizing: 0.20 balances exposure and minimizes fee churn
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag on 1h timeframe

name = "1h_Donchian20_CamarillaR1S1_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Donchian(20) channels (prior completed 4h bar's range)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_ma_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align 4h Donchian levels to 1h timeframe (wait for completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, high_ma_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, low_ma_4h)
    
    # Calculate 1d Camarilla pivot points (R1, S1, PP)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Camarilla calculations: based on prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # R1 = C + (H - L) * 1.1/12
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1/12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 1h timeframe (wait for completed 1d bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 1)  # 4h Donchian needs 20, 1d needs 1
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price below S1 (oversold) AND 4h trend bullish (price > Donchian mid) AND volume spike
            # 4h trend bullish: price above midpoint of Donchian channel
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if (close[i] < s1_aligned[i] and 
                close[i] > donchian_mid and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price above R1 (overbought) AND 4h trend bearish (price < Donchian mid) AND volume spike
            elif (close[i] > r1_aligned[i] and 
                  close[i] < donchian_mid and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses above pivot point (mean reversion complete) OR below Donchian low (trend fails)
            if close[i] > pp_aligned[i] or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses below pivot point (mean reversion complete) OR above Donchian high (trend fails)
            if close[i] < pp_aligned[i] or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals