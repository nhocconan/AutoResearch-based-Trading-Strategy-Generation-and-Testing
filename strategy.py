#!/usr/bin/env python3
# Hypothesis: 6h Williams %R (14) extreme reversion with 1d EMA50 trend filter and 6h volume spike confirmation.
# Long when Williams %R < -80 (oversold) + price > 1d EMA50 (uptrend) + 6h volume > 2.0x 20-period average.
# Short when Williams %R > -20 (overbought) + price < 1d EMA50 (downtrend) + 6h volume > 2.0x 20-period average.
# Exit when Williams %R returns to -50 (mean reversion midpoint).
# Uses discrete position sizing (0.25) to minimize fee churn and volume spike to confirm momentum exhaustion.
# Target: 75-150 total trades over 4 years = 19-37/year for 6h timeframe.
# Works in bull/bear: 1d EMA50 ensures trend alignment, Williams %R captures exhaustion in ranging markets.

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
    
    # --- 6h Williams %R (14-period) ---
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # --- 6h volume confirmation: > 2.0x 20-period average (tight to avoid overtrading) ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA50 calculation
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: price above/below 1d EMA50
    price_above_ema = close > ema_50_1d_aligned
    price_below_ema = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # start after Williams %R warmup
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) + uptrend + volume spike
            if (williams_r[i] < -80 and 
                price_above_ema[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) + downtrend + volume spike
            elif (williams_r[i] > -20 and 
                  price_below_ema[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to -50 (mean reversion)
            if williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to -50 (mean reversion)
            if williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals