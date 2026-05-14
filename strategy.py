#!/usr/bin/env python3
# Hypothesis: 4h Williams %R extreme reversal with 1d EMA34 trend filter and 4h volume spike confirmation.
# Long when Williams %R crosses above -80 from below (oversold bounce) with price > 1d EMA34 (bullish trend) and 4h volume > 1.8x 20-period average.
# Short when Williams %R crosses below -20 from above (overbought rejection) with price < 1d EMA34 (bearish trend) and 4h volume > 1.8x 20-period average.
# Exit on opposite Williams %R level (-20 for longs, -80 for shorts) or trend reversal.
# Uses Williams %R for mean reversion in ranging markets, 1d EMA34 for strong trend filter (reduces whipsaw), and volume spike for conviction.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag.

name = "4h_WilliamsR_Extreme_1dEMA34_4hVolumeSpike"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # Williams %R(14)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 4h volume confirmation: > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (1.8 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) - trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below + price > 1d EMA34 + 4h volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_34_aligned[i] and 
                volume_spike_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from above + price < 1d EMA34 + 4h volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -20 (overbought) or trend turns bearish
            if williams_r[i] >= -20 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -80 (oversold) or trend turns bullish
            if williams_r[i] <= -80 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals