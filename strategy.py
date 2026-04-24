#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator strategy with 1d ATR volatility filter and 1w EMA50 trend filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ATR volatility filter and 1w for EMA50 trend filter.
- Entry: Long when Williams Alligator jaws < teeth < lips (bullish alignment) AND ATR ratio > 1.5 AND price > 1w EMA50.
         Short when Williams Alligator jaws > teeth > lips (bearish alignment) AND ATR ratio > 1.5 AND price < 1w EMA50.
- Exit: Opposite Williams Alligator alignment OR price crosses 1w EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Williams Alligator uses SMAs: jaws=13, teeth=8, lips=5 (all shifted forward).
- ATR ratio (current ATR/20-period ATR) > 1.5 confirms volatility expansion to avoid choppy markets.
- 1w EMA50 provides strong trend filter to avoid counter-trend trades in bear markets.
- Works in bull markets (buy bullish alignment in uptrend) and bear markets (sell bearish alignment in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on trend alignment frequency with strict filters.
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

def williams_alligator(high, low, close):
    """Calculate Williams Alligator: jaws (13), teeth (8), lips (5) SMAs."""
    # Typical price
    typical_price = (high + low + close) / 3.0
    
    # Jaws: 13-period SMA shifted by 8 bars
    jaws = sma(typical_price, 13)
    jaws = np.roll(jaws, 8)
    jaws[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth: 8-period SMA shifted by 5 bars
    teeth = sma(typical_price, 8)
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips: 5-period SMA shifted by 3 bars
    lips = sma(typical_price, 5)
    lips = np.roll(lips, 3)
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    return jaws, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    ema50_1w = sma(df_1w['close'].values, 50)  # Using SMA for stability
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate Williams Alligator on 4h data
    jaws, teeth, lips = williams_alligator(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Williams Alligator conditions
        bullish_alignment = jaws[i] < teeth[i] < lips[i]
        bearish_alignment = jaws[i] > teeth[i] > lips[i]
        
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
        
        # Entry conditions: Williams Alligator alignment with volatility confirmation and trend filter
        if position == 0:
            # Long: bullish alignment AND ATR ratio > 1.5 AND price > 1w EMA50
            if bullish_alignment and atr_ratio_aligned[i] > 1.5 and curr_close > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND ATR ratio > 1.5 AND price < 1w EMA50
            elif bearish_alignment and atr_ratio_aligned[i] > 1.5 and curr_close < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dATR_VolumeFilter_1wEMA50_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0