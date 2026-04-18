#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 3-bar reversal pattern with 1w RSI filter and volume confirmation.
# 3-bar reversal: bullish (close > prev close for 3 consecutive bars) or bearish (close < prev close for 3 consecutive bars).
# 1w RSI < 30 for long, > 70 for short to catch overextended moves in strong trends.
# Volume > 1.5x 20-period average confirms conviction.
# Works in bull markets (catching pullbacks in uptrends) and bear markets (counter-trend bounces).
# Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
name = "1d_3BarReversal_1wRSI_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for price action
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate RSI on 1w data
    close_1w = pd.Series(df_1w['close'].values)
    delta = close_1w.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_values = rsi_1w.values
    
    # Align RSI to lower timeframe (1d)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    # Calculate volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Check for 3-bar reversal pattern
        # Bullish: 3 consecutive higher closes
        bullish_reversal = (close[i] > close[i-1] and 
                           close[i-1] > close[i-2] and 
                           close[i-2] > close[i-3])
        # Bearish: 3 consecutive lower closes
        bearish_reversal = (close[i] < close[i-1] and 
                           close[i-1] < close[i-2] and 
                           close[i-2] < close[i-3])
        
        if position == 0:
            # Long: Bullish reversal AND RSI oversold (<30) AND volume confirmed
            if bullish_reversal and rsi_1w_aligned[i] < 30 and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish reversal AND RSI overbought (>70) AND volume confirmed
            elif bearish_reversal and rsi_1w_aligned[i] > 70 and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish reversal OR RSI overbought (>70)
            if bearish_reversal or rsi_1w_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish reversal OR RSI oversold (<30)
            if bullish_reversal or rsi_1w_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals