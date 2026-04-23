#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Donchian channel breakout with volume confirmation and ATR trailing stop.
Long when price breaks above 1d Donchian upper band (20-period) AND volume > 1.5x 20-period average.
Short when price breaks below 1d Donchian lower band (20-period) AND volume > 1.5x 20-period average.
Exit when price retraces to 1d Donchian middle band or ATR trailing stop hit (2.5*ATR from extreme).
Uses discrete position sizing (0.25) to balance return and drawdown.
Designed for 4h timeframe to target 19-50 trades/year per symbol (75-200 total over 4 years).
Donchian channels from 1d timeframe provide strong structural support/resistance that works in both bull and bear markets.
Volume confirmation filters false breakouts, while ATR trailing stop manages risk and locks in profits.
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
    
    # Calculate 1d Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels: 20-period high/low
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Volume average (20-period) on 4h timeframe
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
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        mid = donchian_mid_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1d Donchian upper band AND volume spike
            if (price > upper and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                long_stop = price - 2.5 * atr_val  # Initial stop
            # Short: Price breaks below 1d Donchian lower band AND volume spike
            elif (price < lower and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                short_stop = price + 2.5 * atr_val  # Initial stop
        else:
            # Update trailing stop and check exit conditions
            exit_signal = False
            
            if position == 1:
                # Update long trailing stop: only move up
                long_stop = max(long_stop, price - 2.5 * atr_val)
                # Exit if price hits trailing stop or retracement to mid band
                if price <= long_stop or price <= mid:
                    exit_signal = True
            else:  # position == -1
                # Update short trailing stop: only move down
                short_stop = min(short_stop, price + 2.5 * atr_val)
                # Exit if price hits trailing stop or retracement to mid band
                if price >= short_stop or price >= mid:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                long_stop = 0.0
                short_stop = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_VolumeConfirmation_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0