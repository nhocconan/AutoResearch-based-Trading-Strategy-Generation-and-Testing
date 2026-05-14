#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extreme with 1d EMA(50) trend filter and 6h volume spike filter.
# Long when Williams %R < -80 (oversold) with 1d EMA(50) bullish (close > EMA) and 6h volume > 2.0x 20-period average.
# Short when Williams %R > -20 (overbought) with 1d EMA(50) bearish (close < EMA) and 6h volume > 2.0x 20-period average.
# Exit when Williams %R returns to neutral range (-50 to -50) or opposite extreme is reached.
# Uses discrete position sizing (0.25) to minimize fee churn and volume spike filter to reduce false signals.
# Williams %R identifies exhaustion points in ranging markets, while EMA(50) ensures trend alignment.
# Volume spike confirms momentum behind the reversal. Works in bull/bear: EMA filter avoids counter-trend trades.

name = "6h_WilliamsR_Extreme_1dEMA50_6hVolumeSpike"
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
    # 6h volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_6h = volume > (2.0 * vol_ma_20)
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, -50)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if missing data
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike_6h[i]) or
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) + 1d EMA bullish + 6h volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) + 1d EMA bearish + 6h volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns above -50 or reaches overbought
            if williams_r[i] > -50 or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns below -50 or reaches oversold
            if williams_r[i] < -50 or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals