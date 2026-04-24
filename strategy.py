#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray combination for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year).
- Primary timeframe: 6h with Williams Alligator (jaw/teeth/lips) for trend direction and Elder Ray (Bull/Bear Power) for momentum confirmation.
- HTF: 12h for trend alignment using EMA34 to avoid counter-trend trades.
- Williams Alligator: SMAs of median price with specific periods (jaw=13, teeth=8, lips=5) to identify trend strength and direction.
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) to measure buying/selling pressure.
- Entry: Long when Alligator is bullish (lips > teeth > jaw) AND Bull Power > 0 AND price > 12h EMA34.
         Short when Alligator is bearish (lips < teeth < jaw) AND Bear Power < 0 AND price < 12h EMA34.
- Exit: Opposite Alligator alignment (when trend weakens/reverses).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams Alligator excels in identifying trending vs ranging markets - only trades when clear trend exists.
- Elder Ray adds momentum confirmation to avoid entering weak trends.
- 12h EMA34 filter ensures alignment with higher timeframe trend to avoid major counter-trend moves.
- Estimated trades: ~100 total over 4 years (~25/year) based on Alligator trend frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def sma(values, period):
    """Calculate Simple Moving Average with proper min_periods."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate median price for Alligator: (high + low) / 2
    median_price = (high + low) / 2
    
    # Williams Alligator components (using median price)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    # Note: Using SMA as approximation for SMMA with proper alignment
    jaw = sma(median_price, 13)
    teeth = sma(median_price, 8)
    lips = sma(median_price, 5)
    
    # Calculate Elder Ray components
    ema13 = ema(close, 13)
    bull_power = high - ema13  # Bull Power = High - EMA(13)
    bear_power = low - ema13   # Bear Power = Low - EMA(13)
    
    # Calculate 12h trend filter: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(15, 40)  # Need 15 for Alligator (max shift), 40 for 12h EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator trend direction
        # Bullish: lips > teeth > jaw
        # Bearish: lips < teeth < jaw
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray momentum
        bull_momentum = bull_power[i] > 0
        bear_momentum = bear_power[i] < 0
        
        # 12h trend filter
        uptrend_12h = close[i] > ema34_12h_aligned[i]
        downtrend_12h = close[i] < ema34_12h_aligned[i]
        
        # Exit conditions: opposite Alligator alignment or loss of momentum
        if position != 0:
            # Exit long: Alligator turns bearish OR loss of bull momentum
            if position == 1:
                if not alligator_bullish or not bull_momentum:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Alligator turns bullish OR loss of bear momentum
            elif position == -1:
                if not alligator_bearish or not bear_momentum:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment + Elder Ray momentum + 12h trend filter
        if position == 0:
            # Long: Alligator bullish AND Bull Power positive AND 12h uptrend
            if alligator_bullish and bull_momentum and uptrend_12h:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power negative AND 12h downtrend
            elif alligator_bearish and bear_momentum and downtrend_12h:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_12hEMA34_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0