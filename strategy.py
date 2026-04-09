#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_spike_v6
# Hypothesis: 4h strategy using 1d Camarilla pivot levels with tighter entry conditions to reduce trade frequency and improve edge.
# Long: Price breaks above H4 with volume > 2.5x 20-period average, RSI(14) > 55, and bullish candle.
# Short: Price breaks below L4 with volume > 2.5x 20-period average, RSI(14) < 45, and bearish candle.
# Exit: Price returns to opposite Camarilla level (H3 for longs, L3 for shorts).
# Tightened volume threshold (2.5x vs 2.0x) and RSI thresholds (55/45 vs 50/50) to reduce false signals.
# Position size: 0.25 (25% of capital) to balance risk and return.
# Target: 15-30 trades/year to minimize fee drag while maintaining statistical significance.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_spike_v6"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # RSI(14) for momentum filter
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    h4_1d = pivot_1d + (range_1d * 1.1 / 2)
    l4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align 1d Camarilla levels to 4h
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(rsi_values[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(open_prices[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.5x 20-period average (tighter)
        volume_confirmed = volume[i] > 2.5 * volume_ma[i]
        # Momentum filter: RSI > 55 for long, < 45 for short (tighter)
        rsi_long_filter = rsi_values[i] > 55
        rsi_short_filter = rsi_values[i] < 45
        # Bullish candle: close > open
        bullish_candle = close[i] > open_prices[i]
        # Bearish candle: close < open
        bearish_candle = close[i] < open_prices[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to H3
            if close[i] <= h3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to L3
            if close[i] >= l3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above H4 with volume, momentum, and bullish candle
            if (close[i] > h4_1d_aligned[i] and    # Break above H4
                volume_confirmed and               # Volume spike (2.5x)
                rsi_long_filter and                # RSI > 55 (strong bullish momentum)
                bullish_candle):                   # Bullish candle
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L4 with volume, momentum, and bearish candle
            elif (close[i] < l4_1d_aligned[i] and  # Break below L4
                  volume_confirmed and             # Volume spike (2.5x)
                  rsi_short_filter and             # RSI < 45 (strong bearish momentum)
                  bearish_candle):                 # Bearish candle
                position = -1
                signals[i] = -0.25
    
    return signals