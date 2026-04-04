#!/usr/bin/env python3
"""
Experiment #2679: 6h Williams %R + 12h ADX trend filter + volume spike
HYPOTHESIS: Williams %R identifies overbought/oversold conditions on 6h, while 12h ADX ensures
we only trade in trending regimes (ADX>25). Volume spike confirms institutional participation.
This mean-reversion-within-trend approach works in both bull (buy pullbacks) and bear (sell rallies)
markets by aligning with the higher timeframe trend. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2679_6h_williamsr_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX trend strength ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX(14)
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smooth(tr, 14)
    plus_dm_14 = wilders_smooth(plus_dm, 14)
    minus_dm_14 = wilders_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di_14 = np.where(tr_14 != 0, (plus_dm_14 / tr_14) * 100, 0)
    minus_di_14 = np.where(tr_14 != 0, (minus_dm_14 / tr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100, 0)
    adx = wilders_smooth(dx, 14)
    
    # Trend strength: ADX > 25 indicates trending market
    adx_trend = adx > 25
    adx_trend_aligned = align_htf_to_ltf(prices, df_12h, adx_trend)
    
    # === 6h Indicators: Williams %R(14), Volume MA(20) ===
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_14 - lowest_14) != 0,
                          ((highest_14 - close) / (highest_14 - lowest_14)) * -100,
                          -50)  # neutral when range=0
    
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
        if (np.isnan(adx_trend_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or Williams %R returns to neutral ---
        if signals[i-1] != 0:  # Currently in position
            current_signal = signals[i-1]
            # Exit if Williams %R returns to oversold/overbought extremes (mean reversion complete)
            if current_signal > 0 and williams_r[i] > -20:  # Long exit
                signals[i] = 0.0
            elif current_signal < 0 and williams_r[i] < -80:  # Short exit
                signals[i] = 0.0
            # Exit if trend weakens (ADX < 20) - avoid whipsaws in ranging markets
            elif not adx_trend_aligned[i]:
                signals[i] = 0.0
            else:
                signals[i] = current_signal  # Hold position
        else:
            # --- New Position Entry Logic ---
            # Require 12h ADX trend filter
            if not adx_trend_aligned[i]:
                signals[i] = 0.0
                continue
            
            # Volume confirmation: require volume spike (> 1.8x average)
            volume_spike = vol_ratio[i] > 1.8
            
            if volume_spike:
                # Long entry: Williams %R oversold (< -80) in uptrend
                if williams_r[i] < -80:
                    signals[i] = SIZE
                # Short entry: Williams %R overbought (> -20) in downtrend
                elif williams_r[i] > -20:
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals