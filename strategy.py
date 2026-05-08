#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_RSI200_Trend_Volume_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily RSI(200) as trend filter - using daily close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/200, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/200, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_200 = 100 - (100 / (1 + rs))
    rsi_200 = rsi_200.values
    
    # Daily volume spike: current volume > 2.0 * 20-day average
    volume_1d = df_1d['volume'].values
    vol_ma20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma20d * 2.0)
    
    # Align daily indicators to 4h timeframe
    rsi_200_aligned = align_htf_to_ltf(prices, df_1d, rsi_200)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # 4-hour Donchian channel breakout (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_200_aligned[i]) or np.isnan(vol_spike_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high with daily RSI > 50 (uptrend) and volume spike
            long_cond = (close[i] > high_20[i] and rsi_200_aligned[i] > 50 and vol_spike_aligned[i])
            
            # Short entry: price breaks below Donchian low with daily RSI < 50 (downtrend) and volume spike
            short_cond = (close[i] < low_20[i] and rsi_200_aligned[i] < 50 and vol_spike_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low (mean reversion)
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high (mean reversion)
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: RSI(200) on daily timeframe provides strong trend filter, combined with 
# Donchian(20) breakout on 4h and volume confirmation. Works in bull markets (breakouts 
# continue with trend) and bear markets (mean reversion at opposite band). 
# RSI(200) is extremely slow, providing reliable trend direction with minimal whipsaw.
# Volume spike (2x 20-day average) ensures momentum confirmation.
# Target: 15-25 trades/year to minimize fee decay while capturing significant moves.