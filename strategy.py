#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 1d EMA200 trend filter and 1d ATR volume spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA200 trend filter and ATR volume confirmation.
- Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs on median price.
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND ATR ratio > 1.5 AND price > 1d EMA200.
         Short when Lips < Teeth < Jaw (bearish alignment) AND ATR ratio > 1.5 AND price < 1d EMA200.
- Exit: When Alligator alignment reverses OR price crosses 1d EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 1.5 confirms volatility expansion to avoid choppy markets.
- 1d EMA200 provides strong trend filter to avoid counter-trend trades.
- Williams Alligator identifies trending vs ranging markets via convergence/divergence.
- Works in bull markets (buy during bullish alignment) and bear markets (sell during bearish alignment).
- Estimated trades: ~80 total over 4 years (~20/year) based on Alligator alignment frequency with filters.
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

def median_price(high, low):
    """Calculate median price (typical price without close)."""
    return (high + low) / 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate Williams Alligator on 6h data (median price)
    med_price = median_price(high, low)
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = sma(med_price, 13)
    teeth = sma(med_price, 8)
    lips = sma(med_price, 5)
    
    # Shift jaw and teeth by their offset values (Alligator specific)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    # Lips typically not shifted or shifted by 3, but we'll use as-is for simplicity
    
    # Handle NaN from rolling and rolling shift
    jaw[:13+8] = np.nan
    teeth[:8+5] = np.nan
    lips[:5] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: Alligator alignment reverses OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Check Alligator alignment
            bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
            bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Exit long: bearish alignment OR price falls below 1d EMA200
            if position == 1:
                if not bullish_align or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish alignment OR price rises above 1d EMA200
            elif position == -1:
                if not bearish_align or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with volatility confirmation and trend filter
        if position == 0:
            # Check Alligator alignment
            bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
            bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: bullish alignment AND ATR ratio > 1.5 AND bullish 1d trend
            if bullish_align and atr_ratio_aligned[i] > 1.5 and curr_close > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND ATR ratio > 1.5 AND bearish 1d trend
            elif bearish_align and atr_ratio_aligned[i] > 1.5 and curr_close < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA200_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0