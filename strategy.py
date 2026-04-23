#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike with HTF Trend Filter
- Long when: Alligator bullish (jaw < teeth < lips) AND Elder Bull Power > 0 AND volume > 2.0x 20-period average AND price > 1d EMA34
- Short when: Alligator bearish (jaw > teeth > lips) AND Elder Bear Power < 0 AND volume > 2.0x 20-period average AND price < 1d EMA34
- Exit when: Alligator reverses (jaws cross teeth) OR Elder Power crosses zero
- Uses 1d EMA34 for HTF trend alignment to avoid counter-trend entries
- Volume spike filter reduces false signals and trade frequency
- Williams Alligator identifies trend initiation, Elder Ray measures bull/bear power
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
- Target: 19-50 trades/year (75-200 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Alligator: SMAs of median price with different periods
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, shifted 8
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, shifted 5
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, shifted 3
    
    # Elder Ray Index: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 13)  # Need 34 for EMA34, 20 for volume MA, 13 for EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions
        alligator_bullish = jaw[i] < teeth[i] and teeth[i] < lips[i]  # Jaw < Teeth < Lips
        alligator_bearish = jaw[i] > teeth[i] and teeth[i] > lips[i]  # Jaw > Teeth > Lips
        
        # Elder Ray conditions
        elder_bull = bull_power[i] > 0
        elder_bear = bear_power[i] < 0
        
        # Trend filter (using 1d EMA34)
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Alligator bullish + Elder bull + uptrend + volume confirmation
            if alligator_bullish and elder_bull and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + Elder bear + downtrend + volume confirmation
            elif alligator_bearish and elder_bear and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator reversal OR Elder Power crosses zero
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator turns bearish OR Elder Bull Power <= 0
                if not alligator_bullish or bull_power[i] <= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short: Alligator turns bullish OR Elder Bear Power >= 0
                if not alligator_bearish or bear_power[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_Volume_1dEMA34_Trend"
timeframe = "4h"
leverage = 1.0