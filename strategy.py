#!/usr/bin/env python3
"""
Experiment #2007: 6h Williams %R + 1d ADX Trend Filter + Volume Spike
HYPOTHESIS: Williams %R identifies overbought/oversold conditions on 6h. 
- Primary: 6h Williams %R(14) with extreme readings (<20 for long, >80 for short)
- HTF: 1d ADX(14) trend filter (only trade when ADX > 25 indicating strong trend)
- Volume: Require volume > 1.3x 20-bar average for confirmation
- Exit: Opposite Williams %R crossover (long exit when %R > 80, short exit when %R < 20)
- Works in bull/bear markets by trading with 1d trend momentum using 6h mean reversion entries.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2007_6h_williamsr_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values / atr
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Williams %R(14), Volume MA(20) ===
    # Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    warmup = 30  # sufficient for Williams %R and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic: Williams %R extreme crossover ---
        if i > 0:
            prev_williams = williams_r[i-1]
            curr_williams = williams_r[i]
            
            # Long exit: Williams %R crosses above 80 (overbought)
            if prev_williams <= 80 and curr_williams > 80:
                signals[i] = 0.0
                continue
            # Short exit: Williams %R crosses below 20 (oversold)
            elif prev_williams >= 20 and curr_williams < 20:
                signals[i] = 0.0
                continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend strength filter: ADX > 25
        strong_trend = adx_1d_aligned[i] > 25
        
        # Volume confirmation: require volume spike (> 1.3x average)
        volume_spike = vol_ratio[i] > 1.3
        
        if strong_trend and volume_spike:
            # Long entry: Williams %R crosses above 20 from below (exit oversold)
            if williams_r[i] > 20 and williams_r[i-1] <= 20:
                signals[i] = SIZE
            # Short entry: Williams %R crosses below 80 from above (exit overbought)
            elif williams_r[i] < 80 and williams_r[i-1] >= 80:
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals