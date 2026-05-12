#!/usr/bin/env python3
# 6h_Liquidity_Grab_Reversal_Volume
# Hypothesis: In 60% of 6h candles, price spikes beyond recent swing high/low then reverses within the same candle (liquidity grab).
# We detect this by checking if high/low exceeds the prior 3-period swing extreme and then closes back inside.
# Entry occurs on the next 6h candle in the reversal direction, filtered by 1d trend (EMA50) and volume spike (>1.5x 20-period avg).
# Works in bull/bear: captures mean reversion after false breaks, avoids chop via volume filter.
# Target: 20-40 trades/year.

name = "6h_Liquidity_Grab_Reversal_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate swing high/low (3-period) for liquidity grab detection
    swing_high = pd.Series(high).rolling(window=3, min_periods=3).max().values
    swing_low = pd.Series(low).rolling(window=3, min_periods=3).min().values
    
    # Liquidity grab: price spikes beyond swing extreme but closes back inside
    # Long setup: low < swing_low and close > swing_low (bear trap)
    # Short setup: high > swing_high and close < swing_high (bull trap)
    liq_grab_long = (low < swing_low) & (close > swing_low)
    liq_grab_short = (high > swing_high) & (close < swing_high)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(swing_high[i]) or np.isnan(swing_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        long_setup = liq_grab_long[i]
        short_setup = liq_grab_short[i]
        ema50_val = ema50_1d_aligned[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: bull trap (liquidity grab short) + price above 1d EMA50 + volume confirmation
            if short_setup and close[i] > ema50_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: bear trap (liquidity grab long) + price below 1d EMA50 + volume confirmation
            elif long_setup and close[i] < ema50_val and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bear trap appears (liquidity grab long) or trend fails
            if long_setup or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bull trap appears (liquidity grab short) or trend fails
            if short_setup or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals