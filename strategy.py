#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h trend filter and 1d volume spike confirmation
# - Uses 4h EMA(50) for trend direction (long when price > EMA50, short when price < EMA50)
# - Uses 1h EMA(9)/EMA(21) crossover for precise entry timing
# - Requires 1d volume > 2.0 * 20-day volume average for confirmation (avoids low-volume breakouts)
# - Includes session filter (08-20 UTC) to trade only during active market hours
# - Fixed position size of 0.20 to control risk and minimize fee churn
# - Target: 15-35 trades/year (~60-140 total over 4 years) to avoid fee drag on 1h timeframe

name = "1h_4h_1d_ema_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume spike confirmation: volume > 2.0 * 20-day average
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Pre-compute 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h EMA(9) and EMA(21) for entry timing
    ema_9 = pd.Series(close).ewm(span=9, min_periods=9, adjust=False).mean().values
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: EMA cross down or trend change
            if ema_9[i] < ema_21[i] or close[i] <= ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions: EMA cross up or trend change
            if ema_9[i] > ema_21[i] or close[i] >= ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for EMA crossover entries with trend and volume confirmation
            if ema_9[i] > ema_21[i] and close[i] > ema_50_4h_aligned[i] and volume_spike_1d_aligned[i]:
                position = 1
                signals[i] = 0.20
            elif ema_9[i] < ema_21[i] and close[i] < ema_50_4h_aligned[i] and volume_spike_1d_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals