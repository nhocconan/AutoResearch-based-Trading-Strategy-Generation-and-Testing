#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla Pivot Reversal with Volume Spike and Choppiness Filter.
Long when price touches S1 with bullish reversal candle and volume spike in choppy market.
Short when price touches R1 with bearish reversal candle and volume spike in choppy market.
Camarilla levels provide intraday support/resistance; volume spike confirms interest;
chop filter (CMW > 61.8) ensures we only mean-revert in ranging markets, avoiding trends.
Works in both bull and bear markets by fading extremes in ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    # Typical Price = (High + Low + Close) / 3
    # Range = High - Low
    # S1 = Close - Range * 1.1 / 12
    # S2 = Close - Range * 1.1 / 6
    # S3 = Close - Range * 1.1 / 4
    # R1 = Close + Range * 1.1 / 12
    # R2 = Close + Range * 1.1 / 6
    # R3 = Close + Range * 1.1 / 4
    
    typical_price = (high + low + close) / 3.0
    range_hl = high - low
    
    # Shift by 1 to use previous day's data
    tp_prev = np.roll(typical_price, 1)
    range_prev = np.roll(range_hl, 1)
    tp_prev[0] = np.nan
    range_prev[0] = np.nan
    
    s1 = tp_prev - range_prev * 1.1 / 12
    r1 = tp_prev + range_prev * 1.1 / 12
    
    # Choppiness Index (EWMA version for efficiency)
    # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    close_prev = np.roll(close, 1)
    tr1 = high - low
    tr2 = np.abs(high - close_prev)
    tr3 = np.abs(low - close_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness = 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hl_range = hh - ll
    chop = np.full_like(tr_sum, 50.0, dtype=float)
    mask = (hl_range > 0) & (~np.isnan(tr_sum)) & (~np.isnan(hl_range))
    chop[mask] = 100 * np.log10(tr_sum[mask] / hl_range[mask]) / np.log10(14)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(s1[i]) or np.isnan(r1[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma_20[i]) or i < 2):  # need i-2 for candle
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Bullish reversal candle: close > open and close > prev close
        bullish_rev = (close[i] > prices['open'].iloc[i]) and (close[i] > close[i-1])
        # Bearish reversal candle: close < open and close < prev close
        bearish_rev = (close[i] < prices['open'].iloc[i]) and (close[i] < close[i-1])
        
        if position == 0:
            # Long: price near S1, bullish reversal, volume spike, choppy market (CHOP > 61.8)
            if (low[i] <= s1[i] * 1.002 and  # allow 0.2% slippage
                bullish_rev and vol_spike and chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price near R1, bearish reversal, volume spike, choppy market
            elif (high[i] >= r1[i] * 0.998 and  # allow 0.2% slippage
                  bearish_rev and vol_spike and chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price touches opposite level or chop drops (trending)
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches R1 or chop < 50 (trending)
                if high[i] >= r1[i] * 0.998 or chop[i] < 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches S1 or chop < 50
                if low[i] <= s1[i] * 1.002 or chop[i] < 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_Pivot_Reversal_Volume_Chop"
timeframe = "4h"
leverage = 1.0