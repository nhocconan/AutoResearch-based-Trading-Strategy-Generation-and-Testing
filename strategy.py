#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with 1d weekly pivot confirmation and volume filter.
# Long when price breaks above 6h Donchian(20) high AND price > 1d weekly pivot R1 AND volume > 1.5x 20-period 6h average.
# Short when price breaks below 6h Donchian(20) low AND price < 1d weekly pivot S1 AND volume > 1.5x 20-period 6h average.
# Exit when price re-enters the Donchian channel (between 20-period high and low) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture breakouts aligned with weekly pivot levels, avoiding false breakouts in ranging markets.
# Weekly pivot levels calculated from 1d OHLC: Pivot = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H.
# Works in both bull and bear markets by requiring volume confirmation and pivot alignment, reducing whipsaws.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_6h)
    
    # === 6h Indicators: ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1d Indicators: Weekly Pivot Levels (R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly pivot: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for Donchian/ATR)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_spike[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(atr_6h[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price re-enters Donchian channel (below 20-period high)
            if price < highest_20[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price re-enters Donchian channel (above 20-period low)
            if price > lowest_20[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND price > weekly R1 AND volume spike
            if price > highest_20[i] and price > r1 and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian low AND price < weekly S1 AND volume spike
            elif price < lowest_20[i] and price < s1 and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_1dWeeklyPivot_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0