#!/usr/bin/env python3
"""
6h_rsi_divergence_1d_volatility_filter_v1
Hypothesis: On 6-hour timeframe, detect RSI divergence (bullish/bearish) with price for reversal signals, filtered by 1-day volatility regime (ATR ratio) to avoid false signals in chop. Bullish divergence: price makes lower low, RSI makes higher low. Bearish divergence: price makes higher high, RSI makes lower high. Enter on close confirmation with volatility expansion (current ATR > 1.2 * ATR(20)). This captures exhaustion moves in both bull and bear markets with low trade frequency (~20-50/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_rsi_divergence_1d_volatility_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volatility filter (ATR)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ATR to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6-hour RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Track recent extrema for divergence detection
    lookback = 10  # bars to look back for swing points
    
    for i in range(max(30, lookback*2), n):
        # Skip if ATR not available
        if np.isnan(atr_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volatility filter: only trade when volatility is expanding (avoid chop)
        vol_expansion = atr_1d_aligned[i] > 1.2 * atr_1d_aligned[i-1]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or bearish divergence
            if rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                # Check for bearish divergence: price higher high, RSI lower high
                lookback_start = max(0, i - lookback)
                if lookback_start < i:
                    price_high = np.max(high[lookback_start:i+1])
                    rsi_high = np.max(rsi[lookback_start:i+1])
                    # Current bar is new high in both price and RSI?
                    if high[i] == price_high and rsi[i] == rsi_high:
                        # Check if this is a divergence: price made HH but RSI made LH vs prior
                        # Find prior swing high
                        prior_high_start = max(0, lookback_start - lookback)
                        if prior_high_start < lookback_start:
                            prior_price_high = np.max(high[prior_high_start:lookback_start])
                            prior_rsi_high = np.max(rsi[prior_high_start:lookback_start])
                            if high[i] > prior_price_high and rsi[i] < prior_rsi_high:
                                position = 0
                                signals[i] = 0.0
            if position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or bullish divergence
            if rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                # Check for bullish divergence: price lower low, RSI higher low
                lookback_start = max(0, i - lookback)
                if lookback_start < i:
                    price_low = np.min(low[lookback_start:i+1])
                    rsi_low = np.min(rsi[lookback_start:i+1])
                    # Current bar is new low in both price and RSI?
                    if low[i] == price_low and rsi[i] == rsi_low:
                        # Check if this is a divergence: price made LL but RSI made HL vs prior
                        prior_low_start = max(0, lookback_start - lookback)
                        if prior_low_start < lookback_start:
                            prior_price_low = np.min(low[prior_low_start:lookback_start])
                            prior_rsi_low = np.min(rsi[prior_low_start:lookback_start])
                            if low[i] < prior_price_low and rsi[i] > prior_rsi_low:
                                position = 0
                                signals[i] = 0.0
            if position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_expansion:
                # Bullish divergence: price makes lower low, RSI makes higher low
                lookback_start = max(0, i - lookback)
                if lookback_start < i:
                    price_low = np.min(low[lookback_start:i+1])
                    rsi_low = np.min(rsi[lookback_start:i+1])
                    # Current bar is new low in both?
                    if low[i] == price_low and rsi[i] == rsi_low:
                        # Check vs prior swing low
                        prior_low_start = max(0, lookback_start - lookback)
                        if prior_low_start < lookback_start:
                            prior_price_low = np.min(low[prior_low_start:lookback_start])
                            prior_rsi_low = np.min(rsi[prior_low_start:lookback_start])
                            if low[i] < prior_price_low and rsi[i] > prior_rsi_low:
                                # Bullish divergence confirmed
                                position = 1
                                signals[i] = 0.25
                # Bearish divergence: price makes higher high, RSI makes lower high
                lookback_start = max(0, i - lookback)
                if lookback_start < i:
                    price_high = np.max(high[lookback_start:i+1])
                    rsi_high = np.max(rsi[lookback_start:i+1])
                    # Current bar is new high in both?
                    if high[i] == price_high and rsi[i] == rsi_high:
                        # Check vs prior swing high
                        prior_high_start = max(0, lookback_start - lookback)
                        if prior_high_start < lookback_start:
                            prior_price_high = np.max(high[prior_high_start:lookback_start])
                            prior_rsi_high = np.max(rsi[prior_high_start:lookback_start])
                            if high[i] > prior_price_high and rsi[i] < prior_rsi_high:
                                # Bearish divergence confirmed
                                position = -1
                                signals[i] = -0.25
    
    return signals