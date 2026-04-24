#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator breakout with 1d EMA200 trend filter and ATR volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA200 trend filter and ATR-based volume spike filter.
- Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs.
- Entry: Long when Alligator lines are bullish (Lips > Teeth > Jaw) AND price breaks above highest line AND ATR(1)/ATR(20) > 1.8 AND price > 1d EMA200.
         Short when Alligator lines are bearish (Lips < Teeth < Jaw) AND price breaks below lowest line AND ATR(1)/ATR(20) > 1.8 AND price < 1d EMA200.
- Exit: Opposite Alligator breakout OR price crosses 1d EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams Alligator identifies trend absence (alligator sleeping) vs presence (alligator awake with mouth open).
- ATR ratio > 1.8 confirms significant volatility expansion to avoid false breakouts.
- 1d EMA200 provides strong trend filter to avoid counter-trend trades and adapt to bull/bear regimes.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with strict filters.
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

def atr(high, low, close, period):
    """Calculate Average True Range with proper min_periods."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def alligator(high, low, close):
    """Calculate Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs of median price."""
    median_price = (high + low) / 2.0
    jaw = sma(median_price, 13)  # 13-period SMA
    jaw = sma(jaw, 8)            # smoothed with 8-period SMA
    teeth = sma(median_price, 8) # 8-period SMA
    teeth = sma(teeth, 5)        # smoothed with 5-period SMA
    lips = sma(median_price, 5)  # 5-period SMA
    lips = sma(lips, 3)          # smoothed with 3-period SMA
    return jaw, teeth, lips

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
    if len(df_1d) < 210:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Williams Alligator from 12h data
    jaw, teeth, lips = alligator(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(210, 20)  # Need sufficient data for EMA200 and Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Alligator breakout OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Exit long: price breaks below Alligator Jaw OR price falls below 1d EMA200
            if position == 1:
                if curr_close < jaw[i] or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Alligator Lips OR price rises above 1d EMA200
            elif position == -1:
                if curr_close > lips[i] or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with volatility confirmation and trend filter
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw (Alligator awake, mouth open up)
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish alignment: Lips < Teeth < Jaw (Alligator awake, mouth open down)
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: bullish Alligator AND price breaks above Alligator Lips AND ATR ratio > 1.8 AND bullish 1d trend
            if bullish_alignment and curr_close > lips[i] and atr_ratio_aligned[i] > 1.8 and curr_close > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator AND price breaks below Alligator Jaw AND ATR ratio > 1.8 AND bearish 1d trend
            elif bearish_alignment and curr_close < jaw[i] and atr_ratio_aligned[i] > 1.8 and curr_close < ema200_1d_aligned[i]:
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