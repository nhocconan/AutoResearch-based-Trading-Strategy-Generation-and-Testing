#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extreme reversal with 1w EMA(50) trend filter and 6h volume spike filter.
# Long when Williams %R < -80 (oversold) with 1w EMA(50) bullish (close > EMA) and 6h volume > 2.0x 20-period average.
# Short when Williams %R > -20 (overbought) with 1w EMA(50) bearish (close < EMA) and 6h volume > 2.0x 20-period average.
# Exit when Williams %R crosses above -50 for longs or below -50 for shorts.
# Uses discrete position sizing (0.25) to minimize fee churn and volume spike filter to reduce false signals.
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe.
# Works in bull/bear: 1w EMA ensures trend alignment, Williams %R captures exhaustion, volume spike confirms momentum.

name = "6h_WilliamsR_Extreme_1wEMA50_6hVolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # --- 6h Indicators (LTF) ---
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 6h volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_6h = volume > (2.0 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) + 1w EMA bullish (close > EMA) + 6h volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) + 1w EMA bearish (close < EMA) + 6h volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (recovering from oversold)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (declining from overbought)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals