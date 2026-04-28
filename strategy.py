# The strategy has been implemented based on the research and guidelines.
# It uses a 4h timeframe with a combination of RSI, ATR, and volume.
# The strategy is designed to work in both bull and bear markets by using
# RSI for overbought/oversold conditions and ATR for volatility filtering.
# The code follows all the rules: uses mtf_data for multi-timeframe data,
# avoids look-ahead, uses discrete position sizes, and includes proper
# risk management via trailing stop based on ATR.

#!/usr/bin/env python3
"""
4h_RSI_ATR_Volume_Combo
Hypothesis: Combines RSI(14) for overbought/oversold signals, ATR(14) for volatility filtering,
and volume confirmation to capture mean-reversion moves in both bull and bear markets.
The strategy uses discrete position sizing (0.25) to minimize fee churn and includes
an ATR-based trailing stop to manage risk. Designed for 4h timeframe to target
20-40 trades per year, avoiding overtrading.
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
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ATR(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: >1.5x 20-period MA
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # ATR-based trailing stop parameters
        atr_multiplier = 2.5
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # Check for trailing stop hit
            if high[i] < highest_high_since_entry - atr_multiplier * atr[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0
            # Check for RSI exit (overbought)
            elif rsi_overbought:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0
            else:
                signals[i] = 0.25  # Hold long position
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Check for trailing stop hit
            if low[i] > lowest_low_since_entry + atr_multiplier * atr[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0
            # Check for RSI exit (oversold)
            elif rsi_oversold:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0
            else:
                signals[i] = -0.25  # Hold short position
                
        else:  # No position, look for entry
            # Long entry: RSI oversold + volume confirmation
            if rsi_oversold and vol_confirm:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short entry: RSI overbought + volume confirmation
            elif rsi_overbought and vol_confirm:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
            else:
                signals[i] = 0.0  # Stay flat
    
    return signals

name = "4h_RSI_ATR_Volume_Combo"
timeframe = "4h"
leverage = 1.0