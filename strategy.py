#!/usr/bin/env python3
"""
6h_MultiTimeframe_RSI_Divergence_Trend
Hypothesis: Combines 6h RSI divergence with 12h EMA trend and volume confirmation to capture momentum reversals.
Works in both bull and bear by using RSI for reversal signals and higher timeframe trend for direction.
Target: 20-40 trades/year (80-160 total over 4 years) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour data for trend and RSI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA for trend filter
    ema_12h = pd.Series(df_12h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 12h RSI for momentum
    delta = pd.Series(df_12h['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h.values)
    
    # 6h volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    bars_since_entry = 0
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        ema_trend = ema_12h_aligned[i]
        rsi = rsi_12h_aligned[i]
        vol_ok = volume_filter[i]
        
        # Detect RSI divergence (simplified: look for RSI extremum with price continuation)
        if i >= 5:
            # Bearish divergence: price makes higher high, RSI makes lower high
            if (high[i] > high[i-3] and high[i-3] > high[i-6] and
                rsi < rsi_12h_aligned[i-3] and rsi_12h_aligned[i-3] < rsi_12h_aligned[i-6]):
                div_signal = -1  # Bearish divergence
            # Bullish divergence: price makes lower low, RSI makes higher low
            elif (low[i] < low[i-3] and low[i-3] < low[i-6] and
                  rsi > rsi_12h_aligned[i-3] and rsi_12h_aligned[i-3] > rsi_12h_aligned[i-6]):
                div_signal = 1   # Bullish divergence
            else:
                div_signal = 0
        else:
            div_signal = 0
        
        if position == 0:
            # Long: bullish divergence with uptrend and volume
            if div_signal == 1 and price > ema_trend and vol_ok:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: bearish divergence with downtrend and volume
            elif div_signal == -1 and price < ema_trend and vol_ok:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            bars_since_entry += 1
            # Minimum holding: 3 bars (1.5 days)
            if bars_since_entry < 3:
                signals[i] = 0.25
            else:
                signals[i] = 0.25
                # Exit: RSI overbought or trend breaks
                if rsi > 70 or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding: 3 bars (1.5 days)
            if bars_since_entry < 3:
                signals[i] = -0.25
            else:
                signals[i] = -0.25
                # Exit: RSI oversold or trend breaks
                if rsi < 30 or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "6h_MultiTimeframe_RSI_Divergence_Trend"
timeframe = "6h"
leverage = 1.0