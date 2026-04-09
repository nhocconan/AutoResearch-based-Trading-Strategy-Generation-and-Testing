#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend + volume confirmation (1.5x 20-period avg)
# Donchian breakouts capture strong momentum moves; 12h EMA50 ensures alignment with higher timeframe trend
# Volume confirmation filters weak breakouts; discrete sizing 0.25 to minimize fee churn
# Works in bull/bear markets: EMA50 trend filter avoids counter-trend entries
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_12h_donchian_ema50_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend direction
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data ONCE before loop for Donchian channel calculation (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Donchian channels from 1d OHLC (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high_20 = np.full(len(df_1d), np.nan)
    donchian_low_20 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i >= 20:
            donchian_high_20[i] = np.max(high_1d[i-20:i])
            donchian_low_20[i] = np.min(low_1d[i-20:i])
        else:
            donchian_high_20[i] = np.nan
            donchian_low_20[i] = np.nan
    
    # Align to 4h timeframe (completed 1d bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR price < 12h EMA50 (trend change)
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR price > 12h EMA50 (trend change)
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + EMA50 trend filter
            if volume_confirmed:
                # Long entry: price > Donchian high AND price > 12h EMA50 (bullish breakout + uptrend)
                if close[i] > donchian_high_aligned[i] and close[i] > ema_50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND price < 12h EMA50 (bearish breakout + downtrend)
                elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals