#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index (14) regime filter + 4h ATR(14) breakout + volume confirmation
# Choppiness Index > 61.8 = ranging market (mean reversion); < 38.2 = trending (trend follow)
# In ranging markets: buy at ATR-based support (low - ATR*1.5), sell at resistance (high + ATR*1.5)
# In trending markets: buy on breakouts above ATR-based resistance, sell on breakdowns below support
# Volume > 1.3x 20-period EMA confirms breakout/breakdown participation
# Target: 25-40 trades/year with regime-adaptive logic for both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR (14-period) for breakout levels and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Choppiness Index (14-period) for regime detection
    # CHOP = 100 * log10(sum(ATR over period) / (max(high) - min(low))) / log10(period)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) > 0, chop_raw, 50.0)  # Default to neutral when range=0
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(14, n):
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # ATR-based support and resistance levels
        support = low[i] - 1.5 * atr[i]
        resistance = high[i] + 1.5 * atr[i]
        
        if position == 0:  # No position - look for entries based on regime
            # Ranging market (CHOP > 61.8): mean reversion at support/resistance
            if chop[i] > 61.8:
                # Long at support with volume confirmation
                if close[i] <= support and volume_confirm:
                    position = 1
                    signals[i] = position_size
                # Short at resistance with volume confirmation
                elif close[i] >= resistance and volume_confirm:
                    position = -1
                    signals[i] = -position_size
            # Trending market (CHOP < 38.2): breakout/breakdown follow-through
            elif chop[i] < 38.2:
                # Long on breakout above resistance with volume
                if close[i] > resistance and volume_confirm:
                    position = 1
                    signals[i] = position_size
                # Short on breakdown below support with volume
                elif close[i] < support and volume_confirm:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit conditions
            # Exit long: price crosses below midpoint OR opposite signal in same regime
            midpoint = (support + resistance) / 2
            if close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            # Reverse to short in trending market on breakdown
            elif chop[i] < 38.2 and close[i] < support and volume_confirm:
                position = -1
                signals[i] = -position_size
        elif position == -1:  # Short position - exit conditions
            # Exit short: price crosses above midpoint OR opposite signal in same regime
            midpoint = (support + resistance) / 2
            if close[i] > midpoint:
                position = 0
                signals[i] = 0.0
            # Reverse to long in trending market on breakout
            elif chop[i] < 38.2 and close[i] > resistance and volume_confirm:
                position = 1
                signals[i] = position_size
    
    return signals

name = "4h_Chop_ATR_Breakout_Volume"
timeframe = "4h"
leverage = 1.0