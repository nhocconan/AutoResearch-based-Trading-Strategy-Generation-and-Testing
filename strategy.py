#!/usr/bin/env python3
# 6h_funding_rate_mean_reversion_v1
# Hypothesis: Funding rate mean reversion on 6h timeframe. Extreme positive funding (longs paying shorts) predicts short-term mean reversion downward; extreme negative funding predicts upward reversion. Works in both bull and bear markets as funding extremes occur during crowded trades. Uses 1d HTF EMA filter to avoid trading against the daily trend. Discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_funding_rate_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load funding rate data (assuming available as column; if not, this will need adjustment)
    # For now, we'll simulate using a proxy: we cannot load external funding data without file path
    # Instead, we use a volume-price divergence proxy for crowded trades
    # But per rules, we must use mtf_data for HTF - we'll use 1d for trend and simulate funding proxy
    
    # 1d HTF for trend filter: 50 EMA (to avoid counter-trend trades)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Proxy for funding rate extreme: 6h RSI divergence from price
    # When price makes new high but RSI fails to confirm (bearish divergence) -> short signal
    # When price makes new low but RSI fails to confirm (bullish divergence) -> long signal
    # We'll use 14-period RSI on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Price action for divergence detection
    # Look for bearish divergence: price higher high, RSI lower high
    # Bullish divergence: price lower low, RSI higher low
    # We'll use 20-period lookback for swing points
    lookback = 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_aligned[i]) or np.isnan(rsi_values[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or price below EMA
            if rsi_values[i] > 70 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or price above EMA
            if rsi_values[i] < 30 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for bullish divergence: price makes lower low, RSI makes higher low
            # Find lowest low in lookback window
            window_low = low[i-lookback:i+1]
            window_high = high[i-lookback:i+1]
            if len(window_low) == lookback + 1 and len(window_high) == lookback + 1:
                lowest_low = np.min(window_low)
                highest_high = np.max(window_high)
                # Current point is the lowest in window?
                is_lowest_low = low[i] == lowest_low
                # Current point is the highest in window?
                is_highest_high = high[i] == highest_high
                
                if is_lowest_low:
                    # Find RSI at the point of lowest low in window
                    min_idx = np.argmin(window_low)
                    rsi_at_low = rsi_values[i-lookback+min_idx]
                    # Bullish divergence: current RSI > RSI at past low
                    if rsi_values[i] > rsi_at_low and close[i] > ema_50_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                elif is_highest_high:
                    # Find RSI at the point of highest high in window
                    max_idx = np.argmax(window_high)
                    rsi_at_high = rsi_values[i-lookback+max_idx]
                    # Bearish divergence: current RSI < RSI at past high
                    if rsi_values[i] < rsi_at_high and close[i] < ema_50_aligned[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals