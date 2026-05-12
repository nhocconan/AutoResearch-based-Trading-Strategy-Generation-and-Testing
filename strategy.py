#!/usr/bin/env python3
name = "1h_Donchian_Breakout_4hTrend_1dVolumeSpike"
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
    
    # === 4h Donchian breakout (20-bar) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    lookback = 20
    upper_4h = np.full(len(high_4h), np.nan)
    lower_4h = np.full(len(low_4h), np.nan)
    for i in range(lookback, len(high_4h)):
        upper_4h[i] = np.max(high_4h[i-lookback:i])
        lower_4h[i] = np.min(low_4h[i-lookback:i])
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # === 4h Trend filter: EMA50 (must be above for long, below for short) ===
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 1d Volume spike filter ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # === Session filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_4h_aligned[i]) or 
            np.isnan(lower_4h_aligned[i]) or
            np.isnan(ema50_4h_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if in_session[i]:
            if position == 0:
                # Long: Close above upper Donchian + above 4h EMA50 + daily volume spike
                if (close[i] > upper_4h_aligned[i] and
                    close[i] > ema50_4h_aligned[i] and
                    vol_spike_1d_aligned[i] > 0.5):
                    signals[i] = 0.20
                    position = 1
                # Short: Close below lower Donchian + below 4h EMA50 + daily volume spike
                elif (close[i] < lower_4h_aligned[i] and
                      close[i] < ema50_4h_aligned[i] and
                      vol_spike_1d_aligned[i] > 0.5):
                    signals[i] = -0.20
                    position = -1
            elif position == 1:
                # Exit long: Close below lower Donchian or below EMA50
                if close[i] < lower_4h_aligned[i] or close[i] < ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: Close above upper Donchian or above EMA50
                if close[i] > upper_4h_aligned[i] or close[i] > ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # Outside session: close positions
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals