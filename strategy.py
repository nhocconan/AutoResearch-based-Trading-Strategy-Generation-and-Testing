#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Williams Fractal regime filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Williams Fractal regime identification (trend vs range).
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h EMA13).
- Regime Filter: Bullish when price > 1d bearish fractal (higher lows), Bearish when price < 1d bullish fractal (lower highs).
- Entry: Long when Bull Power > 0 AND previous Bull Power <= 0 (bullish reversal) AND bullish regime.
         Short when Bear Power < 0 AND previous Bear Power >= 0 (bearish reversal) AND bearish regime.
- Exit: Opposite Elder Ray signal (long exits when Bear Power < 0, short exits when Bull Power > 0).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets (buying strength) and bear markets (selling weakness) by aligning with 1d structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Get 1d data for Williams Fractal regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 days for fractals
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Align fractals to 6h timeframe with extra delay (fractals need 2-bar confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 5)  # Need 13 for EMA, 5 for fractals
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        prev_bull_power = bull_power[i-1]
        prev_bear_power = bear_power[i-1]
        bearish_fractal_level = bearish_fractal_aligned[i]
        bullish_fractal_level = bullish_fractal_aligned[i]
        
        # Regime conditions: bullish when price > 1d bearish fractal (higher lows)
        #                bearish when price < 1d bullish fractal (lower highs)
        bullish_regime = curr_close > bearish_fractal_level
        bearish_regime = curr_close < bullish_fractal_level
        
        # Elder Ray reversal conditions
        bullish_reversal = curr_bull_power > 0 and prev_bull_power <= 0
        bearish_reversal = curr_bear_power < 0 and prev_bear_power >= 0
        
        # Exit conditions: opposite Elder Ray signal
        if position != 0:
            # Exit long: Bear Power turns negative
            if position == 1:
                if curr_bear_power < 0:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bull Power turns positive
            elif position == -1:
                if curr_bull_power > 0:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Elder Ray reversal with regime filter
        if position == 0:
            # Long: bullish reversal AND bullish regime
            long_condition = bullish_reversal and bullish_regime
            
            # Short: bearish reversal AND bearish regime
            short_condition = bearish_reversal and bearish_regime
            
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

name = "6h_ElderRay_BullBearPower_1dWilliamsFractal_Regime_v1"
timeframe = "6h"
leverage = 1.0