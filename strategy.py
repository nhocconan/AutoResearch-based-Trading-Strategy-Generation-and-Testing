#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour TRIX with daily volume spike and weekly ADX trend filter.
# Long when: TRIX crosses above zero, weekly ADX > 25, volume > 2x 20-period average
# Short when: TRIX crosses below zero, weekly ADX > 25, volume > 2x 20-period average
# Exit when: TRIX crosses back through zero
# TRIX captures momentum with reduced noise, volume confirms breakout strength, ADX filters for trending markets.
# Target: 20-30 trades/year per symbol. Works in bull (buy momentum) and bear (sell momentum).
name = "4h_TRIX_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1-week data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate TRIX on daily close: triple EMA of percent change
    # TRIX = EMA(EMA(EMA(close, period), period), period) 
    # We use period=15 as standard
    period = 15
    ema1 = pd.Series(close_1d).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    # Calculate percent change of triple EMA
    pct_change = ema3.pct_change()
    # TRIX is the EMA of the percent change
    trix = pct_change.ewm(span=period, adjust=False, min_periods=period).mean().values * 100  # scale for readability
    
    # Calculate ADX on weekly data
    # ADX requires +DI and -DI calculation
    period_adx = 14
    # Calculate true range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align indices
    
    # Calculate directional movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(arr, period):
        return pd.Series(arr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    atr = wilder_smooth(tr, period_adx)
    plus_di = 100 * wilder_smooth(plus_dm, period_adx) / atr
    minus_di = 100 * wilder_smooth(minus_dm, period_adx) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, period_adx)
    
    # Align 1D and 1W data to 4H timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period*3, period_adx*3)  # Wait for indicator stabilization
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trix_val = trix_aligned[i]
        adx_val = adx_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: TRIX crosses above zero, strong trend, volume spike
            if (trix_val > 0 and trix_aligned[i-1] <= 0 and 
                adx_val > 25 and vol > 2.0 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below zero, strong trend, volume spike
            elif (trix_val < 0 and trix_aligned[i-1] >= 0 and 
                  adx_val > 25 and vol > 2.0 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses back below zero
            if trix_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses back above zero
            if trix_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals