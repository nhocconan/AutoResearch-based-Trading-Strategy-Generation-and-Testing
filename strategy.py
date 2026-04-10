#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR-based volatility breakout with 1d trend filter and volume confirmation
# - Long when price breaks above highest high of last 20 periods + ATR(10) expansion + 1d uptrend + volume > 2.0x 20-bar avg
# - Short when price breaks below lowest low of last 20 periods + ATR(10) expansion + 1d downtrend + volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR filter ensures breakouts occur during expanding volatility (avoids false breakouts in chop)
# - 1d trend filter aligns with higher timeframe momentum
# - Volume confirmation ensures institutional participation
# - Targets ~20-30 trades/year (80-120 total over 4 years) to avoid fee drag

name = "6h_1d_atr_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume confirmation: > 2.0x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ATR(10) for volatility measurement
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # ATR expansion: current ATR > 1.5x ATR of 10 periods ago
    atr_expansion = np.zeros(n, dtype=bool)
    atr_expansion[10:] = atr[10:] > (1.5 * atr[:-10])
    atr_expansion[:10] = False
    
    # Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above highest high + ATR expansion + 1d uptrend + volume spike
            if (close[i] > highest_high_20[i] and 
                atr_expansion[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below lowest low + ATR expansion + 1d downtrend + volume spike
            elif (close[i] < lowest_low_20[i] and 
                  atr_expansion[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price returns to middle of Donchian channel (mean reversion)
            mid_point = (highest_high_20[i] + lowest_low_20[i]) / 2.0
            if position == 1 and close[i] < mid_point:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > mid_point:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals