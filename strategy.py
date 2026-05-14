#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extreme reversal with 1d EMA50 trend filter and 6h volume spike confirmation.
# Long when Williams %R < -80 (oversold) with price > 1d EMA50 (bullish trend) and 6h volume > 2.0x 20-period average.
# Short when Williams %R > -20 (overbought) with price < 1d EMA50 (bearish trend) and 6h volume > 2.0x 20-period average.
# Exit when Williams %R returns to neutral range (-50 to -50) or opposite extreme is reached.
# Uses Williams %R for mean reversion in extended moves, 1d EMA50 for trend alignment, and volume spike for confirmation.
# Target: 75-150 total trades over 4 years (19-38/year) to avoid fee drag while capturing reversals in both bull and bear markets.

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
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                         ((highest_high - close) / (highest_high - lowest_low)) * -100, 
                         -50)
    
    # 6h volume confirmation: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_6h = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R lookback period
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) + price > 1d EMA50 (bullish) + volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) + price < 1d EMA50 (bearish) + volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to -50 or reaches overbought
            if williams_r[i] >= -50 or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to -50 or reaches oversold
            if williams_r[i] <= -50 or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals