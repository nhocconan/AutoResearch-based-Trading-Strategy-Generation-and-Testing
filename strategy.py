#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d Elder Ray regime filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Elder Ray (Bull/Bear Power) regime detection and 1w for Williams Fractal confirmation.
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price.
- Regime: 1d Elder Ray - Bull Power > 0 and Bear Power < 0 = bull regime; Bull Power < 0 and Bear Power > 0 = bear regime.
- Entry: Long when Alligator is bullish (Lips > Teeth > Jaw) AND bull regime AND price > Lips.
         Short when Alligator is bearish (Lips < Teeth < Jaw) AND bear regime AND price < Lips.
- Exit: Opposite Alligator alignment (Lips crosses Teeth).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets via long signals in bull regime, bear markets via short signals in bear regime.
- Avoids whipsaws by requiring Alligator alignment + regime confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Calculate Williams Alligator on 6h data
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values
    
    # Calculate 1d Elder Ray for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for EMA
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1w Williams Fractal for confirmation (optional filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        # If not enough 1w data, continue without fractal filter
        fractal_bullish = np.ones(n)  # Neutral
        fractal_bearish = np.ones(n)  # Neutral
    else:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        
        # Williams Fractal: bearish = high[n-2] is highest of [n-4:n+1], bullish = low[n-2] is lowest
        bearish_fractal = np.zeros(len(high_1w))
        bullish_fractal = np.zeros(len(low_1w))
        
        for i in range(2, len(high_1w)-2):
            if (high_1w[i] >= high_1w[i-2] and high_1w[i] >= high_1w[i-1] and 
                high_1w[i] >= high_1w[i+1] and high_1w[i] >= high_1w[i+2]):
                bearish_fractal[i] = 1
            if (low_1w[i] <= low_1w[i-2] and low_1w[i] <= low_1w[i-1] and 
                low_1w[i] <= low_1w[i+1] and low_1w[i] <= low_1w[i+2]):
                bullish_fractal[i] = 1
        
        # Align fractals to 6h timeframe with 2-bar delay for confirmation
        bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
        bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
        
        # For regime confirmation: in bull regime, avoid bearish fractals; in bear regime, avoid bullish fractals
        fractal_bullish = 1.0 - bearish_fractal_aligned  # 1 = no bearish fractal, 0 = bearish fractal present
        fractal_bearish = 1.0 - bullish_fractal_aligned  # 1 = no bullish fractal, 0 = bullish fractal present
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20)  # Need 13 for Alligator jaw, 20 for 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_median = median_price[i]
        curr_lips = lips[i]
        
        # Alligator alignment
        alligator_bullish = lips[i] > teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] < jaw[i]
        
        # Elder Ray regime
        bull_regime = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        bear_regime = bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0
        
        # Williams Fractal confirmation (if available)
        if len(df_1w) >= 10:
            bull_fractal_ok = fractal_bullish[i] > 0.5  # No bearish fractal
            bear_fractal_ok = fractal_bearish[i] > 0.5  # No bullish fractal
        else:
            bull_fractal_ok = True
            bear_fractal_ok = True
        
        # Exit conditions: opposite Alligator alignment
        if position != 0:
            # Exit long: Alligator turns bearish or lips < teeth
            if position == 1:
                if not alligator_bullish or curr_lips < teeth[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Alligator turns bullish or lips > teeth
            elif position == -1:
                if not alligator_bearish or curr_lips > teeth[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with regime and fractal filters
        if position == 0:
            # Long: Alligator bullish AND bull regime AND price > lips AND no bearish fractal
            long_condition = (alligator_bullish and 
                            bull_regime and
                            curr_median > curr_lips and
                            bull_fractal_ok)
            
            # Short: Alligator bearish AND bear regime AND price < lips AND no bullish fractal
            short_condition = (alligator_bearish and 
                             bear_regime and
                             curr_median < curr_lips and
                             bear_fractal_ok)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dElderRay_1wFractalConfirm_v1"
timeframe = "6h"
leverage = 1.0