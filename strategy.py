#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 1d EMA34 AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND close < 1d EMA34 AND volume > 1.5x 20-period average.
Exit when price retraces to Donchian midpoint or ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee drag while maintaining profit potential.
Donchian channels provide objective breakout levels with proven effectiveness in trending markets.
1d EMA34 filter ensures alignment with long-term trend, reducing whipsaws in choppy markets.
Volume confirmation ensures institutional participation. Works in both bull and bear markets by 
following the 1d trend direction - breakouts in uptrend, breakdowns in downtrend.
Target trade frequency: 19-50 trades/year per symbol (75-200 total over 4 years) to avoid fee drag.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels (20-period) from 1d OHLC
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian calculation: upper = max(high,20), lower = min(low,20), midpoint = (upper+lower)/2
    donch_upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid_1d = (donch_upper_1d + donch_lower_1d) / 2.0
    
    # Align Donchian levels to 4h timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper_1d)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower_1d)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_1d)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, Donchian needs 20, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or np.isnan(donch_mid_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema34_val = ema34_1d_aligned[i]
        donch_upper = donch_upper_aligned[i]
        donch_lower = donch_lower_aligned[i]
        donch_mid = donch_mid_aligned[i]
        
        if position == 0:
            # Long: Break above Donchian upper AND uptrend (price > EMA34) AND volume confirmation
            if close[i] > donch_upper and close[i] > ema34_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Donchian lower AND downtrend (price < EMA34) AND volume confirmation
            elif close[i] < donch_lower and close[i] < ema34_val and volume[i] > 1.5 * vol_ma_val:
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
            
            # Primary exit: Price retraces to Donchian midpoint
            if position == 1 and close[i] <= donch_mid:
                exit_signal = True
            elif position == -1 and close[i] >= donch_mid:
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

name = "4H_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirmation_MidpointExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0