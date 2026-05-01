#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams Alligator combination
# Elder Ray measures bull/bear power via EMA13 (Bull Power = High - EMA13, Bear Power = EMA13 - Low)
# Williams Alligator (Jaw=TEETH=13, Teeth=8, Lips=5 SMAs) provides trend direction and strength
# Long when Bull Power > 0 AND price > Alligator Teeth (8) AND Bull Power rising
# Short when Bear Power > 0 AND price < Alligator Teeth (8) AND Bear Power rising
# Uses 1d Alligator for structural trend filter, 6h Elder Ray for precise timing
# Designed for low trade frequency: ~12-37 trades/year per symbol with 0.25 sizing
# Works in both bull and bear markets by following the dominant daily trend while capturing 6h momentum

name = "6h_ElderRay_1dAlligator_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d HTF data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    close_1d = df_1d['close'].values
    sma_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    sma_8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    sma_13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    
    # Align Alligator lines to 6h timeframe
    sma_5_aligned = align_htf_to_ltf(prices, df_1d, sma_5)
    sma_8_aligned = align_htf_to_ltf(prices, df_1d, sma_8)
    sma_13_aligned = align_htf_to_ltf(prices, df_1d, sma_13)
    
    # 6h Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power
    bear_power = ema_13 - low   # Bear Power
    
    # Rising power conditions (current > previous)
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    # Handle first bar
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d SMA13 (13 days) + 6h EMA13 (13 bars)
    start_idx = max(13, 13)
    
    for i in range(start_idx, n):
        if (np.isnan(sma_5_aligned[i]) or np.isnan(sma_8_aligned[i]) or 
            np.isnan(sma_13_aligned[i]) or np.isnan(ema_13[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d Alligator: 
        # Bullish when price > Teeth (8) AND Teeth > Jaw (13) - aligned alignment
        # Bearish when price < Teeth (8) AND Teeth < Jaw (13)
        bullish_alligator = (close[i] > sma_8_aligned[i]) and (sma_8_aligned[i] > sma_13_aligned[i])
        bearish_alligator = (close[i] < sma_8_aligned[i]) and (sma_8_aligned[i] < sma_13_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            if bullish_alligator:
                # Long: Bull Power > 0 AND rising
                if bull_power[i] > 0 and bull_power_rising[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_alligator:
                # Short: Bear Power > 0 AND rising
                if bear_power[i] > 0 and bear_power_rising[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop or counter-trend
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 OR price < Alligator Teeth (8) OR bearish Alligator alignment
            if (bull_power[i] <= 0) or (close[i] < sma_8_aligned[i]) or not bullish_alligator:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power <= 0 OR price > Alligator Teeth (8) OR bullish Alligator alignment
            if (bear_power[i] <= 0) or (close[i] > sma_8_aligned[i]) or not bearish_alligator:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals