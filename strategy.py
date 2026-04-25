#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeConfirm
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
Uses discrete sizing (0.30) and strict volume threshold (3.0x) to target ~25 trades/year.
In uptrend (close > 1d EMA50): long upper band breakout. In downtrend: short lower band breakout.
Exit on opposite band touch or trend reversal. Designed for low fee drag and robustness in bull/bear.
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
    
    # Get 12h data for Donchian calculations (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels for each 12h bar (based on previous 20 bars)
    upper_12h = np.full(len(close_12h), np.nan)
    lower_12h = np.full(len(close_12h), np.nan)
    
    for i in range(20, len(close_12h)):
        # Upper band: highest high of previous 20 bars
        upper_12h[i] = np.max(high_12h[i-20:i])
        # Lower band: lowest low of previous 20 bars
        lower_12h[i] = np.min(low_12h[i-20:i])
    
    # Align Donchian levels to original timeframe
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 3.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (3.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime
                # Long: break above upper band with volume spike
                long_signal = (close[i] > upper_12h_aligned[i]) and vol_spike[i]
                # Short: break below lower band only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < lower_12h_aligned[i]) and vol_spike[i] and (volume[i] > (5.0 * vol_ma_20[i]))
            else:  # Downtrend regime
                # Short: break below lower band with volume spike
                short_signal = (close[i] < lower_12h_aligned[i]) and vol_spike[i]
                # Long: break above upper band only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > upper_12h_aligned[i]) and vol_spike[i] and (volume[i] > (5.0 * vol_ma_20[i]))
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit conditions: touch lower band or trend reversal
            exit_signal = (close[i] < lower_12h_aligned[i]) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit conditions: touch upper band or trend reversal
            exit_signal = (close[i] > upper_12h_aligned[i]) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0