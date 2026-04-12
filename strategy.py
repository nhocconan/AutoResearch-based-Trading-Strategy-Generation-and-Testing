#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_camarilla_breakout_v40
# Uses 1-day Camarilla pivot levels (H4/L4) as dynamic support/resistance.
# Long when price breaks above H4 with volume confirmation and ADX > 25 (trending).
# Short when price breaks below L4 with volume confirmation and ADX > 25.
# Exit on opposite breakout or when ADX < 20 (range-bound).
# Designed for 4h timeframe to capture multi-day trends in BTC/ETH.
# Target: 20-40 trades/year per symbol for low friction.
name = "4h_1d_camarilla_breakout_v40"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h4 = close_prev + range_prev * 1.1 / 2
    camarilla_l4 = close_prev - range_prev * 1.1 / 2
    
    # Align to 4h timeframe (already delayed by 1 day due to shift)
    h4_level = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_level = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # ADX for trend strength (14-period)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    # Smooth TR, DM+, DM-
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    # DI+ and DI-
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Trend filters: ADX > 25 for entry, ADX < 20 for exit (hysteresis)
    trend_strong = adx > 25
    trend_weak = adx < 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h4_level[i]) or np.isnan(l4_level[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # Exit conditions: opposite breakout or trend weakening
        if position == 1 and (close[i] < l4_level[i] or trend_weak[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > h4_level[i] or trend_weak[i]):
            position = 0
            signals[i] = 0.0
        # Entry conditions: breakout with volume and strong trend
        elif close[i] > h4_level[i] and vol_confirm[i] and trend_strong[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < l4_level[i] and vol_confirm[i] and trend_strong[i] and position != -1:
            position = -1
            signals[i] = -0.25
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals