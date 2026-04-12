#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_rsi_divergence_volume
# Uses daily RSI divergence with price action to identify reversals in both bull and bear markets.
# Long when price makes lower low but RSI makes higher low (bullish divergence) with volume confirmation.
# Short when price makes higher high but RSI makes lower high (bearish divergence) with volume confirmation.
# Exits when RSI crosses opposite extreme (RSI > 70 for long exit, RSI < 30 for short exit).
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drift.
# Works in trending markets via continuation signals and ranging markets via mean reversion.
# Focus on BTC/ETH as primary targets.

name = "4h_1d_rsi_divergence_volume"
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
    
    # Get daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI (14-period)
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align daily RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Volume confirmation: volume > 1.3 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(rsi_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if i >= 2:
            price_lower_low = low[i] < low[i-1] and low[i-1] < low[i-2]
            rsi_higher_low = rsi_aligned[i] > rsi_aligned[i-1] and rsi_aligned[i-1] > rsi_aligned[i-2]
            bullish_div = price_lower_low and rsi_higher_low and rsi_aligned[i] < 40
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            price_higher_high = high[i] > high[i-1] and high[i-1] > high[i-2]
            rsi_lower_high = rsi_aligned[i] < rsi_aligned[i-1] and rsi_aligned[i-1] < rsi_aligned[i-2]
            bearish_div = price_higher_high and rsi_lower_high and rsi_aligned[i] > 60
            
            # Long signal: bullish divergence
            if bullish_div and position != 1:
                position = 1
                signals[i] = 0.25
            # Short signal: bearish divergence
            elif bearish_div and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit conditions: RSI crosses opposite extreme
            elif position == 1 and rsi_aligned[i] >= 70:
                position = 0
                signals[i] = 0.0
            elif position == -1 and rsi_aligned[i] <= 30:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # Hold current position for first few bars
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals