#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Daily VWAP Reversion with Volume Spike and ATR Filter.
# Long when price pulls back to daily VWAP (support) with volume spike and ATR > 0.5*ATR(50).
# Short when price rallies to daily VWAP (resistance) with volume spike and ATR > 0.5*ATR(50).
# Exit when price moves 1.5*ATR away from VWAP in opposite direction.
# Uses daily VWAP as dynamic support/resistance, effective in both trending and ranging markets.
# Target: 60-120 total trades over 4 years (15-30/year) to avoid fee drag.

name = "12h_DailyVWAP_Reversion_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for VWAP calculation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP: typical price * volume / cumulative volume
    typical_price = (df_d['high'] + df_d['low'] + df_d['close']) / 3
    vwap = (typical_price * df_d['volume']).cumsum() / df_d['volume'].cumsum()
    vwap_prev = vwap.shift(1).values  # Use previous day's VWAP as support/resistance
    
    # Align daily VWAP to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_d, vwap_prev)
    
    # Volume filter: current volume > 2.0x 20-period average (higher threshold for fewer trades)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    # ATR filter: ensure sufficient volatility (ATR > 0.5 * ATR(50))
    atr_period = 14
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.maximum(np.absolute(low - np.roll(close, 1)), tr1)
    tr = np.where(np.arange(len(close)) == 0, high - low, tr2)
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr > (0.5 * atr50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for ATR50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(atr_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price at or below VWAP support with volume spike and sufficient volatility
            long_cond = (close[i] <= vwap_aligned[i] * 1.005) and volume_filter[i] and atr_filter[i]
            # Short: price at or above VWAP resistance with volume spike and sufficient volatility
            short_cond = (close[i] >= vwap_aligned[i] * 0.995) and volume_filter[i] and atr_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price moves 1.5*ATR above VWAP (failed support)
            if close[i] > vwap_aligned[i] + (1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price moves 1.5*ATR below VWAP (failed resistance)
            if close[i] < vwap_aligned[i] - (1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals