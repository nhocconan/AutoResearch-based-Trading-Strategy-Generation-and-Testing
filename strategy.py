#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Chande Kroll Stop with daily volume spike and weekly trend filter
# Works in bull/bear: uses adaptive stops (CKS) to cut losses in downtrends and ride trends in uptrends
# Volume spike confirms breakout strength, weekly trend avoids counter-trend trades
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag
name = "12h_1w_ChandeKrollStop_VolumeTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Daily Chande Kroll Stop (dynamic stop levels) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Average True Range (ATR) - 10 period
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Average True Range of price change - 10 period
    price_change = np.abs(close_1d - np.roll(close_1d, 1))
    price_change[0] = 0
    atr_pc = pd.Series(price_change).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Chande Kroll Stop: long stop = highest high - ATR*X, short stop = lowest low + ATR*X
    # X = ATR of price change / ATR (adaptive multiplier)
    x = np.where(atr > 0, atr_pc / atr, 1.0)
    x = np.clip(x, 1.0, 3.0)  # reasonable bounds
    
    highest_high = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    lowest_low = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    long_stop = highest_high - atr * x
    short_stop = lowest_low + atr * x
    
    # Align CKS stops to 12h timeframe
    long_stop_aligned = align_htf_to_ltf(prices, df_1d, long_stop)
    short_stop_aligned = align_htf_to_ltf(prices, df_1d, short_stop)
    
    # === Weekly Trend Filter: EMA20 > EMA50 for uptrend, < for downtrend ===
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = ema20_1w > ema50_1w
    weekly_downtrend = ema20_1w < ema50_1w
    
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # === Daily Volume Spike: volume > 2.0 * 20-period average ===
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma20_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Get values
        close_val = prices['close'].iloc[i]
        long_stop_val = long_stop_aligned[i]
        short_stop_val = short_stop_aligned[i]
        vol_spike = volume_spike_aligned[i]
        wk_up = weekly_uptrend_aligned[i]
        wk_down = weekly_downtrend_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(long_stop_val) or 
            np.isnan(short_stop_val) or np.isnan(vol_spike) or
            np.isnan(wk_up) or np.isnan(wk_down)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above long stop (not stopped out) + volume spike + weekly uptrend
            if close_val > long_stop_val and vol_spike and wk_up:
                signals[i] = 0.25
                position = 1
            # Short: Price below short stop (not stopped out) + volume spike + weekly downtrend
            elif close_val < short_stop_val and vol_spike and wk_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: Continue if above long stop, else exit
            if close_val > long_stop_val:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short: Continue if below short stop, else exit
            if close_val < short_stop_val:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0
    
    return signals