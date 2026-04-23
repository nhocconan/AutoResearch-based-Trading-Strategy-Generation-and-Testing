#!/usr/bin/env python3
"""
Hypothesis: 1d Weekly Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above weekly Donchian high(20) AND close > weekly EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below weekly Donchian low(20) AND close < weekly EMA50 AND volume > 1.8x 20-period average.
Exit when price retraces to weekly Donchian midpoint or ATR trailing stop hit (2.0*ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
Targets 10-25 trades/year per symbol (40-100 total over 4 years) to avoid fee drag.
Designed for BTC and ETH as primary targets with strict entry conditions using higher timeframe structure.
Weekly Donchian provides strong breakout levels, weekly EMA50 filters counter-trend noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian channels and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly Donchian(20): highest high and lowest low of past 20 weekly bars
    dh_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    dl_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    # Weekly midpoint: (Donchian high + Donchian low) / 2
    dc_mid = (dh_20 + dl_20) / 2.0
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    dh_20_aligned = align_htf_to_ltf(prices, df_1w, dh_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_1w, dl_20)
    dc_mid_aligned = align_htf_to_ltf(prices, df_1w, dc_mid)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on daily timeframe
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
    start_idx = max(50, 20)  # EMA50 needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(dh_20_aligned[i]) or np.isnan(dl_20_aligned[i]) or np.isnan(dc_mid_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        dh_val = dh_20_aligned[i]
        dl_val = dl_20_aligned[i]
        dc_mid_val = dc_mid_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high(20) AND uptrend (close > weekly EMA50) AND volume spike
            if price > dh_val and close[i] > ema50_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Price breaks below weekly Donchian low(20) AND downtrend (close < weekly EMA50) AND volume spike
            elif price < dl_val and close[i] < ema50_val and volume[i] > 1.8 * vol_ma_val:
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
            
            # Primary exit: Price retraces to weekly Donchian midpoint
            if position == 1 and price <= dc_mid_val:
                exit_signal = True
            elif position == -1 and price >= dc_mid_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WeeklyDonchian20_VolumeSpike_ATRTrailingStop_MidpointExit"
timeframe = "1d"
leverage = 1.0