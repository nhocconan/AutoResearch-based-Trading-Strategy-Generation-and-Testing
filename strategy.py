#!/usr/bin/env python3
# 6h_adx_williams_alligator_v1
# Hypothesis: 6h strategy combining ADX trend strength with Williams Alligator for regime filtering.
# Long when ADX > 25 (trending) + price > Alligator Jaw (teeth) + Alligator Lips > Teeth (bullish alignment).
# Short when ADX > 25 + price < Alligator Jaw + Alligator Lips < Teeth (bearish alignment).
# Exit when ADX < 20 (range) or opposing alignment occurs.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture strong trends in both bull and bear markets while avoiding whipsaws in ranging conditions.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_williams_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === LTF Indicators (6h) ===
    # ADX calculation (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Williams Alligator (13,8,5 smoothed with future shift)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # future shift
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # future shift
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # future shift
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # === HTF Regime Filter (12h) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Price relative to Jaw (Alligator's backbone)
        price_above_jaw = close[i] > jaw[i]
        price_below_jaw = close[i] < jaw[i]
        
        # HTF trend filter: price above/below 12h EMA50
        htf_uptrend = close[i] > ema_50_12h_aligned[i]
        htf_downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: ADX < 20 (ranging) OR bearish alignment OR price closes below Jaw
            if adx[i] < 20 or bearish_alignment or not price_above_jaw:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX < 20 (ranging) OR bullish alignment OR price closes above Jaw
            if adx[i] < 20 or bullish_alignment or not price_below_jaw:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for entry: ADX > 25 (trending) + alignment + HTF filter
            if adx[i] > 25 and bullish_alignment and price_above_jaw and htf_uptrend:
                position = 1
                signals[i] = 0.25
            elif adx[i] > 25 and bearish_alignment and price_below_jaw and htf_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals