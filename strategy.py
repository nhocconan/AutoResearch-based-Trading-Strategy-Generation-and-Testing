#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour ADX(14) trend strength filter + daily pivot support/resistance + volume confirmation.
# ADX > 25 identifies trending markets (bull or bear) to avoid whipsaws in ranging periods.
# Daily pivot levels act as dynamic support/resistance: buy near S1/S2 in uptrend, sell near R1/R2 in downtrend.
# Volume > 1.3x 20-period average confirms institutional participation.
# Designed for 20-35 trades/year per symbol to minimize fee drag while capturing strong trends.
# Works in both bull (buy pullbacks to support) and bear (sell rallies to resistance) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for pivot points and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Need at least 30 days for ADX calculation
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily timeframe
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # +DM and -DM
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # DI values
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate daily pivot points (using previous day's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = 2 * pivot - df_1d['low']
    s1 = 2 * pivot - df_1d['high']
    r2 = pivot + (df_1d['high'] - df_1d['low'])
    s2 = pivot - (df_1d['high'] - df_1d['low'])
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: price near support (S1 or S2) in uptrend + volume
            if trending and volume_confirmed:
                # Buy near S1 (primary support) or S2 (secondary support)
                near_support = (abs(close[i] - s1_aligned[i]) < 0.005 * close[i]) or \
                               (abs(close[i] - s2_aligned[i]) < 0.01 * close[i])
                if near_support:
                    position = 1
                    signals[i] = position_size
            # Enter short: price near resistance (R1 or R2) in downtrend + volume
            elif trending and volume_confirmed:
                # Sell near R1 (primary resistance) or R2 (secondary resistance)
                near_resistance = (abs(close[i] - r1_aligned[i]) < 0.005 * close[i]) or \
                                  (abs(close[i] - r2_aligned[i]) < 0.01 * close[i])
                if near_resistance:
                    position = -1
                    signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches pivot or R1, or trend weakens
            if (close[i] > pivot_aligned[i] or 
                close[i] > r1_aligned[i] or 
                adx_1d_aligned[i] < 20):  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches pivot or S1, or trend weakens
            if (close[i] < pivot_aligned[i] or 
                close[i] < s1_aligned[i] or 
                adx_1d_aligned[i] < 20):  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_ADX_Pivot_Volume_v1"
timeframe = "4h"
leverage = 1.0