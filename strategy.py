#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) with EMA34 up and volume > 1.5x average.
# Short when Bear Power < 0 AND Bull Power > 0 with EMA34 down and volume > 1.5x average.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Elder Ray measures bull/bear strength relative to EMA13. EMA34 trend filter ensures we trade with the higher timeframe trend.
# Volume confirmation ensures participation. Works in bull markets via upward strength and in bear markets via downward weakness.

name = "6h_ElderRay_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 and EMA34 on 6h data
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema34 = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Elder Ray: Bull Power = Close - EMA13, Bear Power = Low - EMA13
    bull_power = close - ema13
    bear_power = low - ema13
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    close_1d_s = pd.Series(close_1d)
    ema34_1d = close_1d_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe (wait for 1d bar to close)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after sufficient data for EMA34
        # Skip if any required data is NaN
        if (np.isnan(ema34[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 (price above EMA13 but low below EMA13 shows strength)
            #        AND 1d EMA34 trending up (current > previous) AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                i > 0 and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND Bull Power > 0 (price below EMA13 but high above EMA13 shows weakness)
            #        AND 1d EMA34 trending down (current < previous) AND volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] > 0 and 
                  i > 0 and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power >= 0 (price low at or above EMA13 shows weakening) OR 1d EMA34 turns down
            if (bear_power[i] >= 0) or (i > 0 and ema34_1d_aligned[i] < ema34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power <= 0 (price high at or below EMA13 shows weakening) OR 1d EMA34 turns up
            if (bull_power[i] <= 0) or (i > 0 and ema34_1d_aligned[i] > ema34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals