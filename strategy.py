#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + SuperTrend with 1d ATR-based volatility regime filter.
# Long when ADX > 25 (trending) and price > SuperTrend (uptrend).
# Short when ADX > 25 and price < SuperTrend (downtrend).
# Uses 1d ATR percentile to filter low-volatility chop (avoid false signals).
# Target: 15-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on daily
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 50th percentile of ATR as volatility threshold (median ATR)
    atr_median = pd.Series(atr_14).rolling(window=50, min_periods=50).median().values
    vol_regime = atr_14 > atr_median  # High volatility regime
    
    # Align volatility regime to 6h
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate ADX (14-period) on 6h
    # True Range
    tr1_h = high - low
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = high[0] - low[0]
    tr2_h[0] = np.abs(high[0] - close[0])
    tr3_h[0] = np.abs(low[0] - close[0])
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_6h = pd.Series(tr_h).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_6h
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_6h
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # SuperTrend (ATR=10, multiplier=3.0) on 6h
    atr_multiplier = 3.0
    atr_period = 10
    atr_st = pd.Series(tr_h).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    upper_band = (high + low) / 2 + (atr_multiplier * atr_st)
    lower_band = (high + low) / 2 - (atr_multiplier * atr_st)
    
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, n):
        if close[i] > upper_band[i-1]:
            trend[i] = 1
        elif close[i] < lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            if trend[i] == 1 and lower_band[i] > lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if trend[i] == -1 and upper_band[i] < upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        supertrend[i] = lower_band[i] if trend[i] == 1 else upper_band[i]
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(adx[i]) or np.isnan(supertrend[i]) or 
            np.isnan(vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx[i]
        st_val = supertrend[i]
        price = close[i]
        vol_ok = vol_regime_aligned[i]
        
        # Only trade in high volatility regime and strong trend (ADX > 25)
        if vol_ok and adx_val > 25:
            if position == 0:
                # Long: price above SuperTrend (uptrend)
                if price > st_val:
                    signals[i] = 0.25
                    position = 1
                # Short: price below SuperTrend (downtrend)
                elif price < st_val:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Long exit: price crosses below SuperTrend
                if price < st_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price crosses above SuperTrend
                if price > st_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Low volatility or weak trend: exit any position
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_SuperTrend_VolRegime_Filter_v1"
timeframe = "6h"
leverage = 1.0