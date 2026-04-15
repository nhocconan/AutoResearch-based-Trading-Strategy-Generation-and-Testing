#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Breakout with Volume Spike and RSI Momentum Filter
# Uses Bollinger Bands (20,2) on 4h to identify volatility breakouts. Enters long when price breaks above upper band
# with volume > 2x 20-period average and RSI > 55 (momentum confirmation). Enters short when price breaks below lower band
# with volume > 2x average and RSI < 45. Exits when price returns to middle band (20-period SMA).
# Works in both bull and bear markets by capturing momentum bursts after volatility contractions.
# Target: 80-150 total trades over 4 years (20-38/year) to stay within fee drag limits.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2) on 4h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    middle_band = sma_20  # 20-period SMA
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(middle_band[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(rsi[i])):
            continue
        
        # Long entry: price breaks above upper band + volume spike + RSI > 55
        if (close[i] > upper_band[i] and
            volume[i] > 2.0 * vol_ma_20[i] and
            rsi[i] > 55 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower band + volume spike + RSI < 45
        elif (close[i] < lower_band[i] and
              volume[i] > 2.0 * vol_ma_20[i] and
              rsi[i] < 45 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price returns to middle band (mean reversion)
        elif position == 1 and close[i] < middle_band[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > middle_band[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Bollinger_Breakout_Volume_RSI"
timeframe = "4h"
leverage = 1.0