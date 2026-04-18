#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Filter_v1
Strategy: 4h Camarilla pivot breakout with volume confirmation.
Long: Price breaks above R1 with volume > 1.5x 20-period average, close above R1.
Short: Price breaks below S1 with volume > 1.5x 20-period average, close below S1.
Exit: Opposite breakout or trend reversal (EMA34 crossover).
Designed for 4h timeframe: ~20-30 trades/year per symbol (80-120 total over 4 years).
Uses Camarilla levels from daily timeframe for institutional support/resistance.
Volume filter reduces false breakouts. Works in bull/bear via price action at key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Daily OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (Camarilla uses previous day's range)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first day's previous values to NaN (no prior day)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels for each day
    # R1 = Close + 1.1 * (High - Low) / 12
    # S1 = Close - 1.1 * (High - Low) / 12
    rng = prev_high - prev_low
    R1 = prev_close + 1.1 * rng / 12
    S1 = prev_close - 1.1 * rng / 12
    
    # Daily EMA34 for trend filter (used for exit)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily volume average (20-period)
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need enough for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition (for exit only)
        uptrend = ema_34_aligned[i] > np.roll(ema_34_aligned, 1)[i]  # rising EMA
        downtrend = ema_34_aligned[i] < np.roll(ema_34_aligned, 1)[i]  # falling EMA
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Camarilla breakout conditions
        # Long: price breaks above R1 with volume, close above R1
        buy_signal = high[i] > R1_aligned[i] and close[i] > R1_aligned[i] and vol_confirm
        # Short: price breaks below S1 with volume, close below S1
        sell_signal = low[i] < S1_aligned[i] and close[i] < S1_aligned[i] and vol_confirm
        
        if position == 0:
            # Long entry
            if buy_signal:
                signals[i] = 0.25
                position = 1
            # Short entry
            elif sell_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or trend turns down
            if sell_signal or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or trend turns up
            if buy_signal or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0