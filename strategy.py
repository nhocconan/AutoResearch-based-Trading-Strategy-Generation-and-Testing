#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 12h EMA50 trend filter and 1d ATR volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend filter and 1d for ATR volume spike filter.
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price.
- Entry: Long when Lips > Teeth > Jaw (Alligator bullish alignment) AND ATR ratio > 2.0 AND price > 12h EMA50.
         Short when Lips < Teeth < Jaw (Alligator bearish alignment) AND ATR ratio > 2.0 AND price < 12h EMA50.
- Exit: Opposite Alligator alignment OR price crosses 12h EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 2.0 confirms significant volatility expansion to avoid false signals.
- 12h EMA50 provides trend filter to avoid counter-trend trades.
- Williams Alligator identifies trending vs ranging markets (convergence = ranging, divergence = trending).
- Works in bull markets (buy Alligator bullish alignment in uptrend) and bear markets (sell bearish alignment in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Alligator signal frequency with strict filters.
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

def alligator(median_price):
    """Calculate Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3)."""
    jaw = sma(median_price, 13)  # Jaw: 13-period SMA, shifted 8 bars
    teeth = sma(median_price, 8)  # Teeth: 8-period SMA, shifted 5 bars
    lips = sma(median_price, 5)   # Lips: 5-period SMA, shifted 3 bars
    
    # Apply shifts (Alligator is typically shifted forward)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # First values after shift are invalid
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Calculate Williams Alligator
    jaw, teeth, lips = alligator(median_price)
    
    # Calculate 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Exit conditions: opposite Alligator alignment OR price crosses 12h EMA50 in opposite direction
        if position != 0:
            # Exit long: bearish Alligator alignment OR price falls below 12h EMA50
            if position == 1:
                if bearish_alignment or curr_close < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish Alligator alignment OR price rises above 12h EMA50
            elif position == -1:
                if bullish_alignment or curr_close > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with volatility confirmation and trend filter
        if position == 0:
            # Long: bullish Alligator alignment AND ATR ratio > 2.0 AND bullish 12h trend
            if bullish_alignment and atr_ratio_aligned[i] > 2.0 and curr_close > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND ATR ratio > 2.0 AND bearish 12h trend
            elif bearish_alignment and atr_ratio_aligned[i] > 2.0 and curr_close < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_12hEMA50_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0