#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation.
Long when price breaks above Donchian(20) high AND 1d close > 1d EMA34 AND volume > 1.5x 20-period average volume.
Short when price breaks below Donchian(20) low AND 1d close < 1d EMA34 AND volume > 1.5x 20-period average volume.
Exit when price crosses Donchian(10) midpoint OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.30) targeting ~20-50 trades/year on 4h timeframe.
Donchian channels provide clear breakout levels that work in both trending and ranging markets when filtered by higher timeframe trend and volume.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) for breakout signals
    donchian_20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian(10) for exit signals
    donchian_10_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_10_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_10_mid = (donchian_10_high + donchian_10_low) / 2
    
    # 20-period volume average for spike filter
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
    start_idx = max(20, 10, 34)  # donchian20, donchian10, ema34_1d
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_20_high[i]) or np.isnan(donchian_20_low[i]) or
            np.isnan(donchian_10_mid[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema_val = ema_34_1d_aligned[i]
        dc20_high = donchian_20_high[i]
        dc20_low = donchian_20_low[i]
        dc10_mid = donchian_10_mid[i]
        
        if position == 0:
            # Long: Price breaks above Donchian(20) high AND bullish trend (1d close > EMA34) AND volume spike
            if price > dc20_high and close[i] > ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.30
                position = 1
                highest_since_entry = price
            # Short: Price breaks below Donchian(20) low AND bearish trend (1d close < EMA34) AND volume spike
            elif price < dc20_low and close[i] < ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.30
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
            
            # Primary exit: Price crosses Donchian(10) midpoint
            if position == 1 and price < dc10_mid:
                exit_signal = True
            elif position == -1 and price > dc10_mid:
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
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_DC10Exit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0