#!/usr/bin/env python3
"""
4h_1D_CAMARILLA_REVERSION_V1
Hypothesis: Mean reversion at Camarilla pivot levels (H3/L3) on 4h timeframe using 1d pivot calculation.
In ranging markets (common in 2025 BTC/ETH), price tends to revert from extreme daily levels (H3/L3) back toward the daily pivot.
Uses volume confirmation to avoid false signals and RSI to avoid overextended reversals. Works in both bull and bear markets
as mean reversion occurs regardless of trend direction when price reaches statistically extreme levels.
Target: 20-40 trades/year to minimize fee drag while capturing mean reversion edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1D_CAMARILLA_REVERSION_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels - focus on H3/L3 for mean reversion entries
    H3 = pivot + range_hl * 1.1 / 4
    L3 = pivot - range_hl * 1.1 / 4
    H4 = pivot + range_hl * 1.1 / 2  # Stop loss level
    L4 = pivot - range_hl * 1.1 / 2  # Stop loss level
    
    # Align to 4h timeframe
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    H4_4h = align_htf_to_ltf(prices, df_1d, H4)
    L4_4h = align_htf_to_ltf(prices, df_1d, L4)
    
    # RSI(14) for overbought/oversold confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # Volume confirmation (20 period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 1.3x average (moderate filter)
        volume_spike = volume[i] > vol_ma[i] * 1.3
        
        # Mean reversion conditions
        # Long: price at or below L3 with oversold RSI and volume
        long_entry = (close[i] <= L3_4h[i]) and (rsi[i] < 30) and volume_spike
        # Short: price at or above H3 with overbought RSI and volume
        short_entry = (close[i] >= H3_4h[i]) and (rsi[i] > 70) and volume_spike
        
        # Exit conditions: return to pivot or opposite extreme
        long_exit = (close[i] >= pivot[i]) or (close[i] <= L4_4h[i])
        short_exit = (close[i] <= pivot[i]) or (close[i] >= H4_4h[i])
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals