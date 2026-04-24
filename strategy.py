#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + 1d ATR volume spike + 1w EMA50 trend filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1w for EMA50 trend filter and 1d for ATR volume spike filter.
- Entry: Long when Alligator jaw < teeth < lips (bullish alignment) AND ATR ratio > 2.0 AND price > 1w EMA50.
         Short when Alligator jaw > teeth > lips (bearish alignment) AND ATR ratio > 2.0 AND price < 1w EMA50.
- Exit: Opposite Alligator alignment OR price crosses 1w EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 2.0 confirms significant volatility expansion to avoid false breakouts.
- Williams Alligator (SMAs with specific offsets) provides trend direction and alignment confirmation.
- 1w EMA50 provides higher timeframe trend filter to avoid counter-trend trades.
- Works in bull markets (buy in bullish alignment during uptrend) and bear markets (sell in bearish alignment during downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on strict alignment and volatility requirements.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

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

def alligator(high, low, close):
    """Calculate Williams Alligator lines: Jaw, Teeth, Lips."""
    # Jaw: Blue line (13-period SMMA smoothed by 8 bars)
    jaw_raw = sma((high + low) / 2.0, 13)
    jaw = sma(jaw_raw, 8)  # Smoothed by 8 periods
    
    # Teeth: Red line (8-period SMMA smoothed by 5 bars)
    teeth_raw = sma((high + low) / 2.0, 8)
    teeth = sma(teeth_raw, 5)  # Smoothed by 5 periods
    
    # Lips: Green line (5-period SMMA smoothed by 3 bars)
    lips_raw = sma((high + low) / 2.0, 5)
    lips = sma(lips_raw, 3)  # Smoothed by 3 periods
    
    return jaw, teeth, lips

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
    
    # Calculate Williams Alligator on 4h data
    jaw, teeth, lips = alligator(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 100  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Check Alligator alignment
        bullish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
        bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Exit conditions: opposite Alligator alignment OR price crosses 1w EMA50 in opposite direction
        if position != 0:
            # Exit long: bearish alignment OR price falls below 1w EMA50
            if position == 1:
                if bearish_alignment or curr_close < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish alignment OR price rises above 1w EMA50
            elif position == -1:
                if bullish_alignment or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with volatility confirmation and trend filter
        if position == 0:
            # Long: bullish Alligator alignment AND ATR ratio > 2.0 AND bullish 1w trend
            if bullish_alignment and atr_ratio_aligned[i] > 2.0 and curr_close > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND ATR ratio > 2.0 AND bearish 1w trend
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

name = "4h_WilliamsAlligator_1dATR_VolumeSpike_1wEMA50_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0