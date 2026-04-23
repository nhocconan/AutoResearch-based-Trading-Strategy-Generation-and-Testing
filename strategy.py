#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1w trend filter and volume confirmation.
Long when price breaks above Senkou Span A AND price > Kumo cloud AND Tenkan > Kijun AND close > 1w EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below Senkou Span B AND price < Kumo cloud AND Tenkan < Kijun AND close < 1w EMA50 AND volume > 1.5x 20-period average.
Exit when price re-enters Kumo cloud OR ATR trailing stop (2.0*ATR from extreme).
Ichimoku provides dynamic support/resistance and trend direction, effective in both trending and ranging markets.
Weekly EMA50 filter ensures alignment with major trend, reducing counter-trend trades in bear markets.
6h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Ichimoku components from 6h data (no look-ahead)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2.0
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2.0
    
    # Current Kumo (cloud) boundaries: Senkou Span A and B from 26 periods ago
    # To avoid look-ahead, we use values that are already plotted (i.e., from 26 periods ago)
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # Set first 26 values to NaN since we don't have cloud data yet
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    kumo_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
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
    start_idx = max(50, 26, 52, 20, 14)  # EMA50 needs 50, Senkou B needs 52, etc.
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_1w_aligned[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        kumo_top_val = kumo_top[i]
        kumo_bottom_val = kumo_bottom[i]
        
        if position == 0:
            # Long: Price breaks above Kumo top AND bullish TK cross AND above weekly EMA AND volume spike
            if (close[i] > kumo_top_val and 
                tenkan_val > kijun_val and 
                close[i] > ema50_val and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Price breaks below Kumo bottom AND bearish TK cross AND below weekly EMA AND volume spike
            elif (close[i] < kumo_bottom_val and 
                  tenkan_val < kijun_val and 
                  close[i] < ema50_val and 
                  volume[i] > 1.5 * vol_ma_val):
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
            
            # Primary exit: Price re-enters Kumo cloud (between top and bottom)
            if position == 1 and close[i] < kumo_top_val:
                exit_signal = True
            elif position == -1 and close[i] > kumo_bottom_val:
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

name = "6H_Ichimoku_Kumo_Breakout_1wEMA50_Trend_VolumeConfirmation_KumoExit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0