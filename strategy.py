#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1d Williams Fractal regime filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Williams Fractal regime identification (bullish/bearish market structure).
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close).
- Regime Filter: Bullish when 1d bullish fractal confirmed (additional_delay_bars=2), bearish when 1d bearish fractal confirmed.
- Entry Logic: In bullish regime: long when Bull Power > 0 and rising (2-bar momentum).
               In bearish regime: short when Bear Power < 0 and falling (2-bar momentum).
- Exit: Opposite Elder Ray signal (long exits when Bull Power <= 0, short exits when Bear Power >= 0).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets (captures strength via Bull Power) and bear markets (captures weakness via Bear Power).
- Williams Fractal ensures we only trade in the correct structural regime, reducing whipsaws.
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
    
    # Calculate 1d EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d Williams Fractals for regime identification
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Add 2-bar delay for fractal confirmation (needs 2 future 1d bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 13  # Need 13 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        ema_13_level = ema_13_1d_aligned[i]
        bullish_regime = bullish_fractal_aligned[i] == 1
        bearish_regime = bearish_fractal_aligned[i] == 1
        
        # Elder Ray calculations
        bull_power = curr_high - ema_13_level
        bear_power = curr_low - ema_13_level
        
        # Momentum: 2-bar change in power
        if i >= 2:
            prev_bull_power = high[i-2] - ema_13_1d_aligned[i-2]
            prev_bear_power = low[i-2] - ema_13_1d_aligned[i-2]
            bull_power_momentum = bull_power - prev_bull_power
            bear_power_momentum = bear_power - prev_bear_power
        else:
            bull_power_momentum = 0
            bear_power_momentum = 0
        
        # Exit conditions: opposite Elder Ray signal
        if position != 0:
            # Exit long: Bull Power <= 0 (loss of buying pressure)
            if position == 1:
                if bull_power <= 0:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bear Power >= 0 (loss of selling pressure)
            elif position == -1:
                if bear_power >= 0:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Elder Ray with momentum in correct regime
        if position == 0:
            # Long: Bullish regime AND Bull Power > 0 AND rising momentum
            long_condition = bullish_regime and (bull_power > 0) and (bull_power_momentum > 0)
            
            # Short: Bearish regime AND Bear Power < 0 AND falling momentum
            short_condition = bearish_regime and (bear_power < 0) and (bear_power_momentum < 0)
            
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