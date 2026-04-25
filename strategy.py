#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeConfirm
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
Long when price breaks above 20-day high in uptrend (close > weekly EMA50) with volume spike.
Short when price breaks below 20-day low in downtrend (close < weekly EMA50) with volume spike.
Exit when price re-enters the 20-day channel or trend reverses. Designed for low trade frequency (<25/year) and robustness in both bull and bear markets.
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
    
    # Get daily data for Donchian calculations (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian Channel (20-period)
    period20_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian components to original timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, period20_high)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, period20_low)
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (weekly)
                # Long: break above upper band with volume spike
                long_signal = (close[i] > upper_band_aligned[i]) and vol_spike[i]
                # Short: break below lower band only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < lower_band_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # Downtrend regime (weekly)
                # Short: break below lower band with volume spike
                short_signal = (close[i] < lower_band_aligned[i]) and vol_spike[i]
                # Long: break above upper band only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > upper_band_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: re-enter channel or trend reversal
            exit_signal = (close[i] < upper_band_aligned[i]) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: re-enter channel or trend reversal
            exit_signal = (close[i] > lower_band_aligned[i]) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0