#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Donchian upper band AND 1d close > 1d EMA34 AND 12h volume > 2.0x 20-period average volume.
Short when price breaks below Donchian lower band AND 1d close < 1d EMA34 AND 12h volume > 2.0x 20-period average volume.
Exit when price reaches Donchian midpoint OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~12-37 trades/year on 12h timeframe.
Combines price structure (Donchian channels), trend filter (1d EMA34), and volume confirmation for robustness across bull/bear regimes.
Donchian bands calculated from prior 20 completed 12h bars, ensuring no look-ahead bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) from prior completed 12h bars (no look-ahead)
    # Upper band = highest high of prior 20 bars
    # Lower band = lowest low of prior 20 bars
    # Midpoint = (upper + lower) / 2
    high_shifted = np.roll(high, 1)
    low_shifted = np.roll(low, 1)
    high_shifted[0] = np.nan
    low_shifted[0] = np.nan
    
    donchian_upper = pd.Series(high_shifted).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_shifted).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 12h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 12h trailing stop calculation
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
    start_idx = max(20, 34)  # Donchian20 and EMA34 need 20 and 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema_val = ema_34_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        mid = donchian_mid[i]
        
        if position == 0:
            # Long: Break above Donchian upper band AND bullish trend (1d close > EMA34) AND volume spike
            if close[i] > upper and ema_34_1d_aligned[i] > ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Donchian lower band AND bearish trend (1d close < EMA34) AND volume spike
            elif close[i] < lower and ema_34_1d_aligned[i] < ema_val and volume[i] > 2.0 * vol_ma_val:
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
            
            # Primary exit: Price reaches Donchian midpoint
            if position == 1 and close[i] >= mid:
                exit_signal = True
            elif position == -1 and close[i] <= mid:
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

name = "12H_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_MidExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0