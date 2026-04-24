#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA200 trend filter and 1d ATR volume spike.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for Williams Alligator jaw/teeth/lips calculation, 1d for EMA200 trend filter and ATR volume spike.
- Entry: Long when Alligator lips > teeth > jaw (bullish alignment) AND price > 1d EMA200 AND ATR ratio > 1.5.
         Short when Alligator lips < teeth < jaw (bearish alignment) AND price < 1d EMA200 AND ATR ratio > 1.5.
- Exit: Opposite Alligator alignment OR price crosses 1d EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams Alligator uses SMAs of median price: jaw=13, teeth=8, lips=5 periods.
- ATR ratio (current ATR/20-period ATR) > 1.5 confirms volatility expansion to avoid false signals.
- 1d EMA200 provides strong trend filter to avoid counter-trend trades.
- Works in bull markets (buy alignments in uptrend) and bear markets (sell alignments in downtrend).
- Estimated trades: ~80 total over 4 years (~20/year) based on Alligator alignment frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def williams_alligator(high, low, close):
    """Calculate Williams Alligator: jaw(13), teeth(8), lips(5) of median price."""
    median_price = (high + low) / 2.0
    jaw = sma(median_price, 13)  # Blue line
    teeth = sma(median_price, 8)  # Red line
    lips = sma(median_price, 5)   # Green line
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    jaw_1w, teeth_1w, lips_1w = williams_alligator(
        df_1w['high'].values,
        df_1w['low'].values,
        df_1w['close'].values
    )
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w, additional_delay_bars=1)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w, additional_delay_bars=1)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w, additional_delay_bars=1)
    
    # Calculate 1d EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    atr_20_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio_1d = atr_current_1d / (atr_20_1d + 1e-10)  # Avoid division by zero
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 100  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(lips_1w_aligned[i]) or np.isnan(teeth_1w_aligned[i]) or np.isnan(jaw_1w_aligned[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Check Alligator alignment
        bullish_alignment = lips_1w_aligned[i] > teeth_1w_aligned[i] > jaw_1w_aligned[i]
        bearish_alignment = lips_1w_aligned[i] < teeth_1w_aligned[i] < jaw_1w_aligned[i]
        
        # Exit conditions: opposite Alligator alignment OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Exit long: bearish Alligator alignment OR price falls below 1d EMA200
            if position == 1:
                if bearish_alignment or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish Alligator alignment OR price rises above 1d EMA200
            elif position == -1:
                if bullish_alignment or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with volatility confirmation and trend filter
        if position == 0:
            # Long: bullish Alligator alignment AND ATR ratio > 1.5 AND bullish 1d trend
            if bullish_alignment and atr_ratio_1d_aligned[i] > 1.5 and curr_close > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND ATR ratio > 1.5 AND bearish 1d trend
            elif bearish_alignment and atr_ratio_1d_aligned[i] > 1.5 and curr_close < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA200_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0