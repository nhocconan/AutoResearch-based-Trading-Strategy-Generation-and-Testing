#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with daily volume confirmation and EMA trend filter.
# Uses Camarilla levels (H4/L4) from prior day for mean-reversion breakouts.
# Daily volume filter ensures institutional participation.
# 21-period EMA on daily close filters trades in direction of higher timeframe trend.
# Designed for low trade frequency (20-50/year) with clear entry/exit rules.
# Works in bull markets (breakouts above H4 in uptrend) and bear markets (breakouts below L4 in downtrend).
name = "4h_Camarilla_H4L4_Volume_EMA34"
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
    
    # Get daily data for Camarilla levels and EMA (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate prior day's Camarilla levels (H4, L4)
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    camarilla_h4 = close_d + 1.1 * (high_d - low_d) / 2
    camarilla_l4 = close_d - 1.1 * (high_d - low_d) / 2
    
    # Align Camarilla levels to 4h timeframe (use prior day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate daily EMA(34) for trend filter
    ema_34 = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 24-period average volume for confirmation (6 hours worth of 4h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # Long: price breaks above H4 AND volume confirmation AND price above daily EMA34 (uptrend)
            long_breakout = close[i] > camarilla_h4_aligned[i]
            if vol_confirm and long_breakout and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L4 AND volume confirmation AND price below daily EMA34 (downtrend)
            elif vol_confirm and close[i] < camarilla_l4_aligned[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below L4 (mean reversion) OR volume drops below average
            exit_condition = close[i] < camarilla_l4_aligned[i] or volume[i] <= vol_ma_24[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above H4 (mean reversion) OR volume drops below average
            exit_condition = close[i] > camarilla_h4_aligned[i] or volume[i] <= vol_ma_24[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals