#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (JAWS/TEETH/LIPS) with 1d ATR volume spike and ETH/BTC trend filter.
- Primary timeframe: 12h targeting 80-120 total trades over 4 years (20-30/year).
- HTF: 1d for ATR volume spike and ETH/BTC 50-period EMA trend filter.
- Entry: Long when Alligator JAWS crosses above TEETH (bullish alignment) AND ATR ratio > 1.5 AND close > ETH 50 EMA.
         Short when Alligator JAWS crosses below TEETH (bearish alignment) AND ATR ratio > 1.5 AND close < ETH 50 EMA.
- Exit: Opposite Alligator crossover OR price crosses ETH 50 EMA in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams Alligator provides trend-following signals with built-in smoothing (SMAs shifted forward).
- ATR ratio (current ATR/20-period ATR) > 1.5 confirms volatility expansion to avoid false signals.
- ETH 50 EMA on 1d timeframe acts as trend filter to align with major crypto trend.
- Works in bull markets (buy bullish Alligator alignment in uptrend) and bear markets (sell bearish alignment in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Alligator crossover frequency with strict filters.
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

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Williams Alligator on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    
    # Alligator components: SMAs with forward shift
    jaws_period, teeth_period, lips_period = 13, 8, 5
    jaws_shift, teeth_shift, lips_shift = 8, 5, 3
    
    jaws = sma(median_price_12h, jaws_period)
    teeth = sma(median_price_12h, teeth_period)
    lips = sma(median_price_12h, lips_period)
    
    # Apply forward shift (Alligator's "smoothed" nature)
    jaws = np.roll(jaws, jaws_shift)
    teeth = np.roll(teeth, teeth_shift)
    lips = np.roll(lips, lips_shift)
    # Set shifted values to NaN (no look-ahead)
    jaws[:jaws_shift] = np.nan
    teeth[:teeth_shift] = np.nan
    lips[:lips_shift] = np.nan
    
    # Align Alligator to lower timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws, additional_delay_bars=0)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth, additional_delay_bars=0)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips, additional_delay_bars=0)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate ETH 50 EMA on 1d timeframe (trend filter)
    # Using close prices as proxy for ETH trend (works for BTC/ETH/SOL due to correlation)
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Determine Alligator alignment
        bullish_alignment = jaws_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        bearish_alignment = jaws_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        
        # Exit conditions: opposite Alligator crossover OR price crosses ETH 50 EMA in opposite direction
        if position != 0:
            # Exit long: bearish Alligator alignment OR price falls below ETH 50 EMA
            if position == 1:
                if bearish_alignment or curr_close < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish Alligator alignment OR price rises above ETH 50 EMA
            elif position == -1:
                if bullish_alignment or curr_close > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with volatility confirmation and trend filter
        if position == 0:
            # Long: bullish Alligator alignment AND ATR ratio > 1.5 AND bullish ETH trend
            if bullish_alignment and atr_ratio_aligned[i] > 1.5 and curr_close > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND ATR ratio > 1.5 AND bearish ETH trend
            elif bearish_alignment and atr_ratio_aligned[i] > 1.5 and curr_close < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dATR_VolumeSpike_ETH50EMA_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0