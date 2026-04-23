#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above 1d Donchian high AND 1w EMA50 is rising AND volume > 1.5x 20-period average.
Short when price breaks below 1d Donchian low AND 1w EMA50 is falling AND volume > 1.5x 20-period average.
Exit when price retraces to 1d Donchian midpoint.
Uses discrete position sizing (0.25) to minimize fee drag. Targets 7-25 trades/year per symbol.
Donchian provides objective breakout levels; 1w EMA50 ensures alignment with weekly trend; volume confirms breakout strength.
Works in bull (breakouts with volume in uptrend) and bear (breakdowns with volume in downtrend) markets by capturing expansion phases after low volatility.
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
    
    # Calculate 1d OHLC for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) channels from 1d OHLC
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid_1d = (donch_high_1d + donch_low_1d) / 2.0
    
    # Align Donchian levels to 1d timeframe (primary timeframe)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_1d)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate EMA50 slope (rising/falling) - using 3-bar difference for stability
    ema50_slope = np.zeros_like(ema50_1w_aligned)
    ema50_slope[3:] = ema50_1w_aligned[3:] - ema50_1w_aligned[:-3]
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # EMA50 needs 50, Donchian needs 20, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(ema50_slope[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        donch_high = donch_high_aligned[i]
        donch_low = donch_low_aligned[i]
        donch_mid = donch_mid_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        ema50_slope_val = ema50_slope[i]
        
        if position == 0:
            # Long: Break above Donchian high AND rising 1w EMA50 AND volume spike
            if close[i] > donch_high and ema50_slope_val > 0 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND falling 1w EMA50 AND volume spike
            elif close[i] < donch_low and ema50_slope_val < 0 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price retraces to Donchian midpoint
            if position == 1 and close[i] <= donch_mid:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] >= donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirmation_MidExit"
timeframe = "1d"
leverage = 1.0