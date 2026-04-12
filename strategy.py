#!/usr/bin/env python3
"""
4h_12h_Camarilla_Breakout_Volume_Regime_v2
Hypothesis: 4h breakouts at 12h Camarilla H3/L3 levels with volume confirmation and 
12h Choppiness Index regime filter to avoid false breakouts in sideways markets.
Uses 12h for structure (pivots, chop) and 4h for entry timing. Designed for 20-30 
trades/year per symbol with clear trend bias that works in bull (breakouts continue) 
and bear (failed breaks reverse) markets. Regime filter prevents whipsaws in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Breakout_Volume_Regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H DATA FOR CAMARILLA AND REGIME ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12H CHOPPINESS INDEX (14-period) ===
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate true range for 12h
    close_12h_prev = np.roll(close_12h, 1)
    close_12h_prev[0] = close_12h[0]
    tr_12h = true_range(high_12h, low_12h, close_12h_prev)
    
    # Sum of true ranges over 14 periods
    atr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    
    # Absolute price change over 14 periods
    price_change = np.abs(close_12h - np.roll(close_12h, 14))
    price_change[:14] = 0  # Not enough data
    
    # Chop = 100 * log10(sum(tr14) / (atr * n)) / log10(n)
    chop = np.full_like(close_12h, 50.0)  # Default neutral
    for i in range(14, len(close_12h)):
        if atr_14[i] > 0:
            chop[i] = 100 * np.log10(atr_14[i] / price_change[i]) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # === 12H CAMARILLA LEVELS FROM PREVIOUS DAY ===
    # Get daily data for proper OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Map each 12h bar to previous day's OHLC
    pivots_high = np.full(n, np.nan)
    pivots_low = np.full(n, np.nan)
    pivots_close = np.full(n, np.nan)
    
    # Create date lookup for 1d data
    date_to_idx = {}
    for idx in range(len(df_1d)):
        dt = pd.Timestamp(df_1d.iloc[idx]['open_time']).date()
        date_to_idx[dt] = idx
    
    for i in range(n):
        current_date = pd.Timestamp(prices.iloc[i]['open_time']).date()
        prev_date = current_date - pd.Timedelta(days=1)
        
        if prev_date in date_to_idx:
            prev_day_idx = date_to_idx[prev_date]
            ph = df_1d['high'].iloc[prev_day_idx]
            pl = df_1d['low'].iloc[prev_day_idx]
            pc = df_1d['close'].iloc[prev_day_idx]
            
            pivots_high[i] = ph
            pivots_low[i] = pl
            pivots_close[i] = pc
    
    # Calculate Camarilla H3 and L3 levels
    H3 = pivots_close + (pivots_high - pivots_low) * 1.1 / 4
    L3 = pivots_close - (pivots_high - pivots_low) * 1.1 / 4
    
    # Volume filter (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: Chop < 50 = trending (favor breakouts), Chop > 50 = ranging (avoid)
        trending_regime = chop_aligned[i] < 50
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)  # Increased threshold for fewer trades
        
        # Long: price breaks above H3 with volume and trending regime
        long_signal = (close[i] > H3[i] and 
                      strong_volume and 
                      trending_regime)
        
        # Short: price breaks below L3 with volume and trending regime
        short_signal = (close[i] < L3[i] and 
                       strong_volume and 
                       trending_regime)
        
        # Exit: price returns to midpoint or regime changes to choppy
        exit_long = (position == 1 and 
                    (close[i] < pivots_close[i] or chop_aligned[i] > 55))
        exit_short = (position == -1 and 
                     (close[i] > pivots_close[i] or chop_aligned[i] > 55))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals