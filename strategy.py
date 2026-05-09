#!/usr/bin/env python3

# Hypothesis: 1d timeframe with daily Bollinger Bands squeeze and RSI mean reversion.
# Uses Bollinger Band width percentile to detect low volatility (squeeze) conditions,
# then enters mean reversion trades when RSI reaches extreme levels (<30 or >70).
# The squeeze acts as a volatility filter to avoid whipsaw in high volatility periods,
# while RSI extremes provide entry signals. Works in both bull and bear markets
# by capturing mean reversion within ranges and avoiding strong trends.
# Target: 50-100 total trades over 4 years (12-25/year) with size 0.25.

name = "1d_BollingerSqueeze_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0

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
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    bb_width = upper - lower
    
    # Bollinger Band width percentile (50-period lookback) to identify squeeze
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).rank(pct=True).values
    squeeze = bb_width_percentile < 0.2  # Bottom 20% = low volatility squeeze
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI extremes for mean reversion
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    # Volume filter: current volume > 1.5x 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(squeeze[i]) or np.isnan(rsi_oversold[i]) or np.isnan(rsi_overbought[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bollinger squeeze + RSI oversold + volume confirmation
            if squeeze[i] and rsi_oversold[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bollinger squeeze + RSI overbought + volume confirmation
            elif squeeze[i] and rsi_overbought[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or squeeze breaks
            if rsi[i] >= 50 or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or squeeze breaks
            if rsi[i] <= 50 or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals