#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Strategy: Daily Camarilla pivot levels with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (L3, H3) act as strong support/resistance. 
# Price approaching these levels with volume confirmation and aligned with weekly trend
# offers high-probability mean-reversion or breakout opportunities.
# Weekly trend filter avoids counter-trend trades. Designed for low trade frequency
# (~15-25 trades/year) to minimize fee drag in ranging and trending markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly volume average (20-period) for confirmation
    volume_1w = df_1w['volume'].values
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    # Align raw weekly volume for confirmation
    vol_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # Camarilla levels: 
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # We'll use H3 and L3 as entry/exit levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Handle first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate Camarilla levels
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20_1w_aligned[i]) or \
           np.isnan(vol_1w_aligned[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter: price vs weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current weekly volume > 1.3x 20-period average
        vol_confirm = vol_1w_aligned[i] > 1.3 * vol_avg_20_1w_aligned[i]
        
        # Price near Camarilla levels (within 0.5% tolerance)
        near_h3 = abs((close[i] - camarilla_h3[i]) / camarilla_h3[i]) < 0.005
        near_l3 = abs((close[i] - camarilla_l3[i]) / camarilla_l3[i]) < 0.005
        
        # Entry conditions
        # Long: Price near L3 support AND weekly uptrend AND volume confirmation
        if near_l3 and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price near H3 resistance AND weekly downtrend AND volume confirmation
        elif near_h3 and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price moves to opposite Camarilla level or crosses weekly EMA
        elif position == 1 and (near_h3 or close[i] < ema_50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (near_l3 or close[i] > ema_50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals