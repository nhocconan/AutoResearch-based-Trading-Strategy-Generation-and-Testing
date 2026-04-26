#!/usr/bin/env python3
"""
6h_ADX_DMI_Trend_1dRegime_v1
Hypothesis: On 6h timeframe, use ADX(14) + DMI crossover for trend signals, filtered by 1d regime (ADX(14) > 25 for trending, < 20 for ranging). In trending regime, follow 6h DMI crossovers. In ranging regime, fade moves to Bollinger Bands(20,2) with volume confirmation. Uses discrete position sizing (0.25) to limit drawdown. Designed for low trade frequency (target 12-37/year) to overcome fee drag in ranging/bear markets like 2025+. Works in both bull (trend following) and bear (mean reversion in ranges) via regime-adaptive logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d for regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h ADX(14) + DMI for trend signals
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[np.nan], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    
    dm_plus_6h = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                          np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus_6h = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                           np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus_6h = np.concatenate([[0], dm_plus_6h])
    dm_minus_6h = np.concatenate([[0], dm_minus_6h])
    
    tr_14_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14_6h = pd.Series(dm_plus_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14_6h = pd.Series(dm_minus_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    di_plus_6h = 100 * dm_plus_14_6h / tr_14_6h
    di_minus_6h = 100 * dm_minus_14_6h / tr_14_6h
    
    # Calculate Bollinger Bands(20,2) on 6h for ranging regime
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d ADX, 6h ADX/DMI, BB, volume MA
    start_idx = max(14*3, 20, 20) + 1  # ~43 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or
            np.isnan(tr_14_6h[i]) or
            np.isnan(di_plus_6h[i]) or
            np.isnan(di_minus_6h[i]) or
            np.isnan(ma_20[i]) or
            np.isnan(std_20[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        adx_1d = adx_1d_aligned[i]
        
        # Regime determination
        trending_regime = adx_1d > 25
        ranging_regime = adx_1d < 20
        
        if position == 0:
            if trending_regime:
                # Trend following: DMI crossover
                long_signal = (di_plus_6h[i] > di_minus_6h[i]) and vol_conf
                short_signal = (di_minus_6h[i] > di_plus_6h[i]) and vol_conf
            elif ranging_regime:
                # Mean reversion: fade at Bollinger Bands with volume
                long_signal = (close_val < lower_bb[i]) and vol_conf
                short_signal = (close_val > upper_bb[i]) and vol_conf
            else:
                # Transition regime: no signals
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions
            if trending_regime:
                # Exit trend: DMI cross down
                if di_minus_6h[i] > di_plus_6h[i]:
                    signals[i] = 0.0
                    position = 0
            elif ranging_regime:
                # Exit mean reversion: price returns to middle
                if close_val > ma_20[i]:
                    signals[i] = 0.0
                    position = 0
            else:
                # Transition regime: exit on opposite signal
                if (di_minus_6h[i] > di_plus_6h[i]) or (close_val > ma_20[i]):
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions
            if trending_regime:
                # Exit trend: DMI cross up
                if di_plus_6h[i] > di_minus_6h[i]:
                    signals[i] = 0.0
                    position = 0
            elif ranging_regime:
                # Exit mean reversion: price returns to middle
                if close_val < ma_20[i]:
                    signals[i] = 0.0
                    position = 0
            else:
                # Transition regime: exit on opposite signal
                if (di_plus_6h[i] > di_minus_6h[i]) or (close_val < ma_20[i]):
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "6h_ADX_DMI_Trend_1dRegime_v1"
timeframe = "6h"
leverage = 1.0