#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout + 1d volume spike + 1w ADX regime filter (low ADX for ranging markets)
# - Long: price breaks above Donchian upper channel (20-period) + 1d volume > 1.8x 20-period volume average + 1w ADX < 25
# - Short: price breaks below Donchian lower channel (20-period) + 1d volume > 1.8x 20-period volume average + 1w ADX < 25
# - Exit: price returns to Donchian midpoint (mean reversion)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-50 trades/year to stay within fee drag limits
# - ADX filter ensures we avoid strong trending markets where breakouts fail
# - Works in both bull and bear markets by focusing on low-volatility ranging conditions where price respects Donchian levels

name = "4h_1d_1w_donchian_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h data ONCE before loop for Donchian calculation (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Load 1w data ONCE before loop for ADX regime filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper and lower channels
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    # Pre-compute 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1w ADX (14-period) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w).rolling(2).max() - pd.Series(low_1w).rolling(2).min()
    tr2 = abs(pd.Series(high_1w).shift(1) - pd.Series(close_1w))
    tr3 = abs(pd.Series(low_1w).shift(1) - pd.Series(close_1w))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1w).diff()
    down_move = pd.Series(low_1w).diff()
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    sum_up = pd.Series(up_move).rolling(window=14, min_periods=14).sum().values
    sum_down = pd.Series(down_move).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * sum_up / (atr + 1e-10)
    di_minus = 100 * sum_down / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        # Volume confirmation: 1d volume > 1.8x 20-period volume average (tight threshold)
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 1.8 * volume_sma_20_1d_aligned[i]
        
        # Weekly ADX filter: ADX < 25 (low trend/ranging market)
        weekly_adx = adx_aligned[i]
        adx_filter = weekly_adx < 25
        
        # Donchian breakout conditions
        donchian_breakout_long = close_price > donchian_upper_aligned[i]
        donchian_breakout_short = close_price < donchian_lower_aligned[i]
        
        # Entry conditions
        enter_long = donchian_breakout_long and vol_confirm and adx_filter
        enter_short = donchian_breakout_short and vol_confirm and adx_filter
        
        # Exit conditions: price returns to Donchian midpoint (mean reversion)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops below midpoint
            exit_long = close_price < donchian_middle_aligned[i]
        elif position == -1:
            # Exit short if price rises above midpoint
            exit_short = close_price > donchian_middle_aligned[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals