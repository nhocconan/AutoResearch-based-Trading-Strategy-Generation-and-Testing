#!/usr/bin/env python3
"""
Experiment #2091: 6h Williams %R + 1d ADX Trend Filter + Volume Spike
HYPOTHESIS: Williams %R identifies overbought/oversold conditions on 6h, 
while 1d ADX > 25 filters for trending markets only. Volume spikes confirm 
momentum. Works in bull/bear by trading pullbacks in strong trends.
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2091_6h_williamsr_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
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
        tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        plus_dm_sum = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values
        minus_dm_sum = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_sum / tr_sum
        minus_di = 100 * minus_dm_sum / tr_sum
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        
        return adx, plus_di, minus_di
    
    adx_1d, plus_di_1d, minus_di_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    # Trend strength: ADX > 25 indicates strong trend
    trend_strong = adx_1d > 25
    # Trend direction: +DI > -DI for uptrend, vice versa
    trend_up = plus_di_1d > minus_di_1d
    trend_down = plus_di_1d < minus_di_1d
    
    # Combine for trend bias: 1 for strong uptrend, -1 for strong downtrend, 0 otherwise
    trend_bias_1d = np.where(trend_strong & trend_up, 1, 
                            np.where(trend_strong & trend_down, -1, 0))
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # === 6h Indicators: Williams %R(14), Volume MA(20) ===
    # Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                         ((highest_high - close) / (highest_high - lowest_low)) * -100,
                         -50)  # neutral when range is 0
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(trend_bias_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R signals: 
        # Long when oversold (< -80) in uptrend
        # Short when overbought (> -20) in downtrend
        if trend_bias_aligned[i] > 0 and williams_r[i] < -80 and vol_ratio[i] > 1.5:
            signals[i] = SIZE
        elif trend_bias_aligned[i] < 0 and williams_r[i] > -20 and vol_ratio[i] > 1.5:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals