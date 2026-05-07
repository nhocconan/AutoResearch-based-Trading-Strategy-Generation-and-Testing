#!/usr/bin/env python3
"""
12h_Alligator_ElderRay_Trend_Strategy
Hypothesis: On 12h timeframe, combine Williams Alligator (trend direction) with Elder Ray Index (bull/bear power) and volume confirmation to capture strong trends in both bull and bear markets. 
- Williams Alligator: Jaw(13), Teeth(8), Lips(5) SMAs on median price. Bullish when Lips > Teeth > Jaw, bearish when reversed.
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low. Confirms trend strength.
- Volume filter: Current volume > 1.5x 20-period average to avoid low-conviction moves.
- Entry: Go long when Alligator bullish AND Bull Power > 0 AND volume filter. Go short when Alligator bearish AND Bear Power > 0 AND volume filter.
- Exit: When Alligator direction changes or power fails.
- Uses 1d EMA34 as higher timeframe trend filter to avoid counter-trend trades.
- Designed for low trade frequency (~15-25/year) to minimize fee drag while capturing major trends.
"""
name = "12h_Alligator_ElderRay_Trend_Strategy"
timeframe = "12h"
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
    
    # Get daily data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator calculation (using median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8-period SMA
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # 5-period SMA
    
    # Elder Ray Index calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0
    
    start_idx = max(20, 13)  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 12 bars between trades (6 days on 12h TF) to reduce frequency
            if bars_since_exit < 12:
                continue
                
            # Alligator bullish: Lips > Teeth > Jaw
            alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Alligator bearish: Jaw > Teeth > Lips
            alligator_bearish = jaw[i] > teeth[i] and teeth[i] > lips[i]
            
            # Long conditions: Alligator bullish AND Bull Power positive AND volume filter AND price above 1d EMA
            if (alligator_bullish and 
                bull_power[i] > 0 and 
                volume_filter[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short conditions: Alligator bearish AND Bear Power positive AND volume filter AND price below 1d EMA
            elif (alligator_bearish and 
                  bear_power[i] > 0 and 
                  volume_filter[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit conditions: Alligator direction change OR power failure OR trend filter violation
            alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
            alligator_bearish = jaw[i] > teeth[i] and teeth[i] > lips[i]
            
            if position == 1:
                # Exit long if: Alligator turns bearish OR Bull Power fails OR price below 1d EMA
                if (not alligator_bullish or bull_power[i] <= 0 or close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                # Exit short if: Alligator turns bullish OR Bear Power fails OR price above 1d EMA
                if (not alligator_bearish or bear_power[i] <= 0 or close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals