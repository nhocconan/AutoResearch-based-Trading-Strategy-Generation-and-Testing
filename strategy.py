#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d + RSI(14) + volume confirmation
# Enter long when: price > S3 (support) and RSI < 40 (oversold) and volume > 1.5x avg
# Enter short when: price < R3 (resistance) and RSI > 60 (overbought) and volume > 1.5x avg
# Camarilla levels provide institutional support/resistance; RSI avoids false breakouts
# Volume confirms institutional participation. Target: 50-150 trades over 4 years.

name = "6h_camarilla_rsi_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    # Range = high - low
    # Resistance levels: R3 = close + (high - low) * 1.1/2, R4 = close + (high - low) * 1.1
    # Support levels: S3 = close - (high - low) * 1.1/2, S4 = close - (high - low) * 1.1
    range_1d = high_1d - low_1d
    r3 = close_1d + range_1d * 1.1 / 2
    r4 = close_1d + range_1d * 1.1
    s3 = close_1d - range_1d * 1.1 / 2
    s4 = close_1d - range_1d * 1.1
    
    # Align Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(volume_threshold[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price < S4 OR RSI > 60 (overbought)
            if close[i] < s4_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price > R4 OR RSI < 40 (oversold)
            if close[i] > r4_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price at S3/R3 with RSI extreme + volume
            if volume[i] > volume_threshold[i]:
                # Long: price > S3 and RSI < 40 (oversold bounce)
                if close[i] > s3_aligned[i] and rsi[i] < 40:
                    signals[i] = 0.25
                    position = 1
                # Short: price < R3 and RSI > 60 (overbought rejection)
                elif close[i] < r3_aligned[i] and rsi[i] > 60:
                    signals[i] = -0.25
                    position = -1
    
    return signals