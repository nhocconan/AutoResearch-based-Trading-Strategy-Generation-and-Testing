#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using Elder Ray Bull/Bear Power with 12h EMA34 trend filter and volume confirmation.
Long when Bull Power > 0 (close > EMA13) AND price > 12h EMA34 AND volume > 1.5x 20-period average.
Short when Bear Power < 0 (close < EMA13) AND price < 12h EMA34 AND volume > 1.5x 20-period average.
Exit when price crosses EMA13 or ATR trailing stop hit (2.5*ATR from highest/lowest since entry).
Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
Designed for 6h timeframe targeting ~15-25 trades/year per symbol (60-100 total over 4 years).
Focus on BTC and ETH as primary targets with volume confirmation to filter false signals.
Elder Ray captures bull/bear strength relative to EMA13; 12h EMA34 provides higher-timeframe trend filter.
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
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # EMA13 for Elder Ray (primary timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = close - EMA13, Bear Power = close - EMA13 (negative when bearish)
    bull_power = close - ema13  # >0 when bullish
    bear_power = close - ema13  # <0 when bearish (same calculation, we check sign)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 13)  # 12h EMA34 needs 34, vol MA needs 20, EMA13 needs 13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(ema13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema34_val = ema34_12h_aligned[i]
        ema13_val = ema13[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND price > 12h EMA34 AND volume spike
            if bull_val > 0 and price > ema34_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Bear Power < 0 AND price < 12h EMA34 AND volume spike
            elif bear_val < 0 and price < ema34_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price crosses EMA13 (Elder Ray power changes sign)
            if position == 1 and price <= ema13_val:
                exit_signal = True
            elif position == -1 and price >= ema13_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_12hEMA34_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0