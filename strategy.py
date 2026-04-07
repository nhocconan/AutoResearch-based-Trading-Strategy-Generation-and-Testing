#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ehlers Fisher Transform with 1-day ADX regime filter
# Long when Fisher crosses above -1.5 in trending up (ADX>25, +DI>-DI)
# Short when Fisher crosses below +1.5 in trending down (ADX>25, +DI<DI)
# Fisher Transform identifies turning points with Gaussian normalization
# ADX regime filter avoids whipsaws in ranging markets
# Target: 80-160 total trades over 4 years (20-40/year)
# Position size: 0.25 (25% of capital)

name = "6h_fisher_1d_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for ADX and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day ADX (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_s = pd.Series(tr)
    dm_plus_s = pd.Series(dm_plus)
    dm_minus_s = pd.Series(dm_minus)
    
    atr = tr_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_sm = dm_plus_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_sm = dm_minus_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_sm / (atr + 1e-10)
    di_minus = 100 * dm_minus_sm / (atr + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX and DI to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    di_plus_aligned = align_htf_to_ltf(prices, df_1d, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_1d, di_minus)
    
    # 6-hour Ehlers Fisher Transform (price normalized to [-1, 1] over 10 periods)
    # Price position: (price - min) / (max - min) * 2 - 1
    lookback = 10
    highest = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(close).rolling(window=lookback, min_periods=lookback).min().values
    price_norm = 2 * (close - lowest) / (highest - lowest + 1e-10) - 1
    
    # Fisher Transform: 0.5 * ln((1+price_norm)/(1-price_norm))
    # Avoid division by zero
    fisher_raw = 0.5 * np.log((1 + price_norm) / (1 - price_norm + 1e-10))
    # Smooth with 3-period EMA
    fisher = pd.Series(fisher_raw).ewm(span=3, adjust=False).mean().values
    
    # 6-period ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(fisher[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(di_plus_aligned[i]) or np.isnan(di_minus_aligned[i]) or 
            np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Fisher crosses below 0 (mean reversion) or ADX weak (<20)
            elif fisher[i] < 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Fisher crosses above 0 (mean reversion) or ADX weak (<20)
            elif fisher[i] > 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Fisher crossovers with ADX trend filter
            # Trend up: ADX>25 and +DI > -DI
            # Trend down: ADX>25 and +DI < -DI
            trending_up = adx_aligned[i] > 25 and di_plus_aligned[i] > di_minus_aligned[i]
            trending_down = adx_aligned[i] > 25 and di_plus_aligned[i] < di_minus_aligned[i]
            
            # Long: Fisher crosses above -1.5 in uptrend
            if i > 0 and fisher[i] > -1.5 and fisher[i-1] <= -1.5 and trending_up:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Fisher crosses below +1.5 in downtrend
            elif i > 0 and fisher[i] < 1.5 and fisher[i-1] >= 1.5 and trending_down:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals