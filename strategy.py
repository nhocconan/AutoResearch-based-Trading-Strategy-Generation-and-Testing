#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
Strategy: 12-hour Camarilla pivot breakout with volume confirmation and ATR-based stoploss.
Long: Price breaks above R1 + volume > 1.3x average + ATR(12) < 50th percentile ATR(144)
Short: Price breaks below S1 + volume > 1.3x average + ATR(12) < 50th percentile ATR(144)
Exit: Price closes below R1 (long) or above S1 (short) OR ATR spike exit
Position size: 0.25
Designed to capture intraday swings with volatility filter to avoid choppy markets.
Timeframe: 12h
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
    
    # Calculate Camarilla levels from previous 12h bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous bar
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Calculate ATR(12) for volatility filter and stop
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.maximum(np.abs(low - np.roll(close, 1)), tr1)
    tr2[0] = np.nan
    atr_raw = pd.Series(tr2).rolling(window=12, min_periods=12).mean().values
    
    # Calculate ATR(144) for percentile rank (approx 12d of 12h data)
    atr_long_raw = pd.Series(tr2).rolling(window=144, min_periods=144).mean().values
    # Percentile rank of current ATR vs long-term ATR
    atr_percentile = pd.Series(atr_raw).rolling(window=144, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(144, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1[i]) or 
            np.isnan(S1[i]) or 
            np.isnan(atr_percentile[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Volatility filter: only trade when volatility is low (below median)
        vol_filter = atr_percentile[i] < 0.5
        
        # Breakout conditions
        breakout_up = close[i] > R1[i]
        breakout_down = close[i] < S1[i]
        
        # Exit conditions: price re-enters the range
        exit_long = close[i] < R1[i]
        exit_short = close[i] > S1[i]
        
        # ATR spike exit: if volatility spikes above 80th percentile
        vol_spike_exit = atr_percentile[i] > 0.8
        
        if position == 0:
            # Long: breakout above R1 + volume filter + volatility filter
            if breakout_up and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + volatility filter
            elif breakout_down and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below R1 OR volatility spike
            if exit_long or vol_spike_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above S1 OR volatility spike
            if exit_short or vol_spike_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0