#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness index regime filter + 1-day Williams %R mean reversion
# Long when: CHOP(14) > 61.8 (range) AND Williams %R(14) < -80 (oversold) AND price > VWAP(20)
# Short when: CHOP(14) > 61.8 (range) AND Williams %R(14) > -20 (overbought) AND price < VWAP(20)
# Exit when: CHOP(14) < 38.2 (trending) OR Williams %R crosses back above -50 (for longs) / below -50 (for shorts)
# This strategy targets mean reversion in ranging markets while avoiding trending conditions
# Works in both bull/bear by focusing on range-bound behavior which occurs in all regimes
# Target: 50-100 trades/year to minimize fee drag while capturing mean reversion opportunities

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)) * -100
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate VWAP on 4h (20-period)
    typical_price = (high + low + close) / 3
    vwap_numerator = (typical_price * volume).cumsum()
    vwap_denominator = volume.cumsum()
    vwap = vwap_numerator / vwap_denominator
    
    # Calculate Choppiness Index on 4h (14-period)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first bar TR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr.sum() / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop = np.where((highest_high_14 - lowest_low_14) == 0, 50, chop)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: ranging market + oversold + price above VWAP
            if (chop[i] > 61.8 and williams_r_aligned[i] < -80 and price > vwap[i]):
                position = 1
                signals[i] = position_size
            # Short setup: ranging market + overbought + price below VWAP
            elif (chop[i] > 61.8 and williams_r_aligned[i] > -20 and price < vwap[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trending market OR Williams %R crosses above -50
            if chop[i] < 38.2 or williams_r_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trending market OR Williams %R crosses below -50
            if chop[i] < 38.2 or williams_r_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Chop_WilliamsR_VWAP_MeanReversion"
timeframe = "4h"
leverage = 1.0