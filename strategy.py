#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 4-period RSI mean reversion with 1-day Bollinger Band width filter
# Long when RSI < 30 and BB width > 0.03 (high volatility mean reversion)
# Short when RSI > 70 and BB width > 0.03
# Exit when RSI crosses 50
# Uses Bollinger Band width as volatility filter to avoid low volatility whipsaws
# Targets 20-50 trades/year to minimize fee drag
# Works in both bull and bear markets by capturing overextended moves in high volatility regimes

name = "1d_RSI_BBWidth_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI(4)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands(20, 2)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # normalized width
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for BB calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(bb_width[i]) or 
            np.isnan(bb_middle[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        bb_width_val = bb_width[i]
        bb_middle_val = bb_middle[i]
        
        if position == 0:
            # Enter long: RSI < 30 and BB width > 0.03 (oversold in high volatility)
            if rsi_val < 30 and bb_width_val > 0.03:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70 and BB width > 0.03 (overbought in high volatility)
            elif rsi_val > 70 and bb_width_val > 0.03:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 50
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses below 50
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals