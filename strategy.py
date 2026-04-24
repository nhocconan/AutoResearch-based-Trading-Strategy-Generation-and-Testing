#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and 1d ATR volume spike confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend filter and 1d for ATR volume spike filter.
- Entry: Long when Jaw < Teeth < Lips (bullish alignment) AND ATR ratio > 2.0 AND price > 1w EMA50.
         Short when Jaw > Teeth > Lips (bearish alignment) AND ATR ratio > 2.0 AND price < 1w EMA50.
- Exit: Opposite Alligator alignment OR price crosses 1w EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams Alligator identifies trend phases through smoothed moving averages (Jaw=13, Teeth=8, Lips=5).
- 1w EMA50 provides higher timeframe trend filter to avoid counter-trend trades.
- ATR ratio (current ATR/20-period ATR) > 2.0 confirms significant volatility expansion to avoid false signals.
- Works in bull markets (buy in bullish alignment with uptrend) and bear markets (sell in bearish alignment with downtrend).
- Estimated trades: ~60 total over 4 years (~15/year) based on trend persistence with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def smoothed_moving_average(values, period):
    """Calculate Smoothed Moving Average (SMMA) used in Williams Alligator."""
    sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
    smma = np.full_like(values, np.nan, dtype=float)
    if len(values) >= period:
        smma[period-1] = sma[period-1]
        for i in range(period, len(values)):
            smma[i] = (smma[i-1] * (period-1) + values[i]) / period
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Williams Alligator from 1d data: Jaw (13), Teeth (8), Lips (5) - all SMMA
    jaw = smoothed_moving_average(close, 13)
    teeth = smoothed_moving_average(close, 8)
    lips = smoothed_moving_average(close, 5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Alligator alignment OR price crosses 1w EMA50 in opposite direction
        if position != 0:
            # Check Alligator alignment
            jaw_val = jaw[i]
            teeth_val = teeth[i]
            lips_val = lips[i]
            
            # Bullish alignment: Jaw < Teeth < Lips
            bullish_alignment = jaw_val < teeth_val < lips_val
            # Bearish alignment: Jaw > Teeth > Lips
            bearish_alignment = jaw_val > teeth_val > lips_val
            
            if position == 1:
                # Exit long: bearish alignment OR price falls below 1w EMA50
                if bearish_alignment or curr_close < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            elif position == -1:
                # Exit short: bullish alignment OR price rises above 1w EMA50
                if bullish_alignment or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with volatility confirmation and trend filter
        if position == 0:
            # Check Alligator alignment
            jaw_val = jaw[i]
            teeth_val = teeth[i]
            lips_val = lips[i]
            
            # Bullish alignment: Jaw < Teeth < Lips
            bullish_alignment = jaw_val < teeth_val < lips_val
            # Bearish alignment: Jaw > Teeth > Lips
            bearish_alignment = jaw_val > teeth_val > lips_val
            
            # Long: bullish alignment AND ATR ratio > 2.0 AND price > 1w EMA50
            if bullish_alignment and atr_ratio_aligned[i] > 2.0 and curr_close > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND ATR ratio > 2.0 AND price < 1w EMA50
            elif bearish_alignment and atr_ratio_aligned[i] > 2.0 and curr_close < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_1wEMA50_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0