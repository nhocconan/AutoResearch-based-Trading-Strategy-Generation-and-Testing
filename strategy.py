#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 4h EMA trend and volume spike confirmation.
# Long when CHOP > 61.8 (range) AND price > EMA20 AND volume > 2x 20-period average.
# Short when CHOP > 61.8 (range) AND price < EMA20 AND volume > 2x 20-period average.
# Exit when CHOP < 38.2 (trending) or price crosses EMA20 in opposite direction.
# This strategy targets mean reversion in ranging markets with trend alignment and volume confirmation.
# Choppiness Index identifies ranging vs trending markets. EMA20 provides dynamic support/resistance.
# Volume spike confirms institutional participation in the mean reversion play.
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by adapting to ranging conditions.

name = "4h_Choppiness_EMA20_Volume_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Choppiness Index (14-period)
    def calculate_choppiness(high, low, close, period=14):
        atr = np.zeros(len(close))
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        atr = np.zeros(len(close))
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        highest_high = np.zeros(len(close))
        lowest_low = np.zeros(len(close))
        highest_high[period-1] = np.max(high[:period])
        lowest_low[period-1] = np.min(low[:period])
        for i in range(period, len(close)):
            highest_high[i] = max(highest_high[i-1], high[i])
            lowest_low[i] = min(lowest_low[i-1], low[i])
        
        chop = np.full(len(close), 50.0)
        for i in range(period-1, len(close)):
            if highest_high[i] != lowest_low[i]:
                chop[i] = 100 * np.log10(atr[i] * period / (highest_high[i] - lowest_low[i])) / np.log10(period)
        return chop
    
    chop = calculate_choppiness(high, low, close)
    
    # EMA20 for dynamic trend
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Sufficient warmup for chop and EMA20
    
    for i in range(start_idx, n):
        if (np.isnan(chop[i]) or np.isnan(ema20[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: choppy market (CHOP > 61.8) AND price > EMA20 AND volume filter
            long_cond = (chop[i] > 61.8) and (close[i] > ema20[i]) and volume_filter[i]
            # Short conditions: choppy market (CHOP > 61.8) AND price < EMA20 AND volume filter
            short_cond = (chop[i] > 61.8) and (close[i] < ema20[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: chop becomes trending (CHOP < 38.2) OR price crosses below EMA20
            if (chop[i] < 38.2) or (close[i] < ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: chop becomes trending (CHOP < 38.2) OR price crosses above EMA20
            if (chop[i] < 38.2) or (close[i] > ema20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals