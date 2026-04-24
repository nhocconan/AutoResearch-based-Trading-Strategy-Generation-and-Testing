#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray + 1d EMA trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for trend filter (price above/below EMA34).
- Entry: Long when Alligator is bullish (jaw < teeth < lips) AND Elder Ray bull power > 0 AND price > 1d EMA34.
         Short when Alligator is bearish (jaw > teeth > lips) AND Elder Ray bear power < 0 AND price < 1d EMA34.
- Exit: Opposite Alligator alignment OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams Alligator identifies trend absence/presence and direction via smoothed medians.
- Elder Ray measures bull/bear power behind the move.
- Works in bull markets (buy when all bullish aligned) and bear markets (sell when all bearish aligned).
- Estimated trades: ~80 total over 4 years (~20/year) based on Alligator alignment frequency with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def smma(values, period):
    """Calculate Smoothed Moving Average (used in Alligator)."""
    # SMMA is similar to EMA but with different smoothing
    return pd.Series(values).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Williams Alligator on 12h (jaw=13, teeth=8, lips=5 SMMA of median price)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    jaw_12h = smma(median_price_12h, 13)  # Blue line
    teeth_12h = smma(median_price_12h, 8)   # Red line
    lips_12h = smma(median_price_12h, 5)    # Green line
    
    # Align Alligator lines to 12h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h, additional_delay_bars=1)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h, additional_delay_bars=1)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h, additional_delay_bars=1)
    
    # Elder Ray on 12h (bull power = high - EMA13, bear power = low - EMA13)
    ema13_12h = ema(df_12h['close'].values, 13)
    bull_power_12h = df_12h['high'].values - ema13_12h
    bear_power_12h = df_12h['low'].values - ema13_12h
    
    # Align Elder Ray to 12h timeframe
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h, additional_delay_bars=1)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h, additional_delay_bars=1)
    
    # 1d trend filter
    trend_bullish = close > ema34_1d_aligned
    trend_bearish = close < ema34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for Alligator/EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or np.isnan(lips_12h_aligned[i]) or
            np.isnan(bull_power_12h_aligned[i]) or np.isnan(bear_power_12h_aligned[i]) or
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Alligator alignment OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: Alligator turns bearish OR price falls below 1d EMA34
            if position == 1:
                if (jaw_12h_aligned[i] > teeth_12h_aligned[i] and teeth_12h_aligned[i] > lips_12h_aligned[i]) or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Alligator turns bullish OR price rises above 1d EMA34
            elif position == -1:
                if (jaw_12h_aligned[i] < teeth_12h_aligned[i] and teeth_12h_aligned[i] < lips_12h_aligned[i]) or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: All aligned in same direction
        if position == 0:
            # Long: Alligator bullish AND Elder Ray bullish AND bullish 1d trend
            if (jaw_12h_aligned[i] < teeth_12h_aligned[i] and teeth_12h_aligned[i] < lips_12h_aligned[i] and
                bull_power_12h_aligned[i] > 0 and trend_bullish[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Elder Ray bearish AND bearish 1d trend
            elif (jaw_12h_aligned[i] > teeth_12h_aligned[i] and teeth_12h_aligned[i] > lips_12h_aligned[i] and
                  bear_power_12h_aligned[i] < 0 and trend_bearish[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_1dEMA34_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0