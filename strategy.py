#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA50 trend filter + 6h volume spike confirmation.
# Long when Bull Power > 0 (close > EMA13) AND price > 1d EMA50 (bullish trend) AND 6h volume > 2.0x 20-period average.
# Short when Bear Power < 0 (close < EMA13) AND price < 1d EMA50 (bearish trend) AND 6h volume > 2.0x 20-period average.
# Exit when power signal reverses (Bull/Bear Power crosses zero) or volume condition fails.
# Uses Elder Ray to measure trend strength via price-EMA13 relationship, 1d EMA50 for higher timeframe trend alignment,
# and volume spike to confirm institutional participation. Designed to work in both bull (trend following) and bear (counter-trend retracements) markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "6h_ElderRay_BullBearPower_1dEMA50_6hVolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # 6h EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = Close - EMA13, Bear Power = EMA13 - Close
    bull_power = close - ema_13
    bear_power = ema_13 - close
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
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (close > EMA13) + price > 1d EMA50 (bullish trend) + 6h volume spike
            if (bull_power[i] > 0 and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 (close < EMA13) + price < 1d EMA50 (bearish trend) + 6h volume spike
            elif (bear_power[i] > 0 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (close <= EMA13) OR loss of volume confirmation
            if (bull_power[i] <= 0 or not volume_spike_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power <= 0 (close >= EMA13) OR loss of volume confirmation
            if (bear_power[i] <= 0 or not volume_spike_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals