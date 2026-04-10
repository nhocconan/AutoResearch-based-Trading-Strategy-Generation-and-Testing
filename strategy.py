#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation
# - Camarilla pivot levels: calculated from previous 1d OHLC (L3, H3 for short, L4, H4 for long)
# - 1d ADX > 25 trend filter: ensures we trade only when higher timeframe is trending
# - Volume confirmation: current volume > 1.5x 20-period average to avoid false breakouts
# - Exit: Camarilla opposite level touch (L3 for long exit, H3 for short exit)
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Target: 12-37 trades/year on 12h (50-150 total over 4 years) to minimize fee drag

name = "12h_1d_camarilla_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ADX for trend filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff().values
    dm_minus = -pd.Series(low_1d).diff().values
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed TR and DM
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    di_plus_14 = 100 * dm_plus_14 / tr_14
    di_minus_14 = 100 * dm_minus_14 / tr_14
    dx = 100 * abs(di_plus_14 - di_minus_14) / (di_plus_14 + di_minus_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    trend_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 12h Camarilla pivot levels from previous 1d bar
    # We need previous day's OHLC for current 12h bar's pivot levels
    prev_close_1d = pd.Series(close_1d).shift(1).values
    prev_high_1d = pd.Series(high_1d).shift(1).values
    prev_low_1d = pd.Series(low_1d).shift(1).values
    prev_open_1d = df_1d['open'].shift(1).values
    
    # Camarilla pivot levels
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # H2 = close + 0.75*(high-low), L2 = close - 0.75*(high-low)
    # H1 = close + 0.5*(high-low), L1 = close - 0.5*(high-low)
    pivot_high = prev_close_1d
    pivot_range = prev_high_1d - prev_low_1d
    
    H4 = pivot_high + 1.5 * pivot_range
    L4 = pivot_high - 1.5 * pivot_range
    H3 = pivot_high + 1.125 * pivot_range
    L3 = pivot_high - 1.125 * pivot_range
    H2 = pivot_high + 0.75 * pivot_range
    L2 = pivot_high - 0.75 * pivot_range
    H1 = pivot_high + 0.5 * pivot_range
    L1 = pivot_high - 0.5 * pivot_range
    
    # Align pivot levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    
    # Pre-compute 12h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup for ADX and pivots
        # Skip if any required data is invalid
        if (np.isnan(trend_aligned[i]) or np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = trend_aligned[i] > 25
        
        # Entry conditions: Camarilla H4 breakout for long, L4 breakdown for short
        long_breakout = prices['close'].iloc[i] > H4_aligned[i]
        short_breakout = prices['close'].iloc[i] < L4_aligned[i]
        
        # Exit conditions: touch opposite Camarilla level
        exit_long = prices['close'].iloc[i] < L3_aligned[i]  # Touch L3 for long exit
        exit_short = prices['close'].iloc[i] > H3_aligned[i]  # Touch H3 for short exit
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: H4 breakout AND trending AND volume confirmation
            if long_breakout and trending and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short conditions: L4 breakdown AND trending AND volume confirmation
            elif short_breakout and trending and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: touch opposite Camarilla level
            exit_condition = (position == 1 and exit_long) or (position == -1 and exit_short)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals