#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d volume spike + 1w ADX regime filter
# - Long: price breaks above Camarilla H3 level + 1d volume > 1.8x 20-period volume average + 1w ADX < 25 (low trend/ranging)
# - Short: price breaks below Camarilla L3 level + 1d volume > 1.8x 20-period volume average + 1w ADX < 25 (low trend/ranging)
# - Exit: price reverses back to Camarilla pivot point (PP) or touches opposite level (H3/L3)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-50 trades/year to stay within fee drag limits
# - ADX filter ensures we avoid strong trending markets where breakouts fail
# - Works in both bull and bear markets by focusing on low-volatility ranging conditions where price respects Camarilla levels

name = "4h_1d_1w_camarilla_adx_v1"
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
    
    # Load 4h data ONCE before loop for Camarilla calculation (MTF rule compliance)
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
    
    # Pre-compute 4h Camarilla levels (based on previous day's OHLC)
    # Camarilla levels calculated from daily OHLC, applied to 4h timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels for today based on yesterday's OHLC
    # H4 = close + 1.5*(high-low), H3 = close + 1.25*(high-low), etc.
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    camarilla_r = high_1d - low_1d
    
    camarilla_h3 = camarilla_pp + 1.125 * camarilla_r  # H3 = PP + 1.125*range
    camarilla_l3 = camarilla_pp - 1.125 * camarilla_r  # L3 = PP - 1.125*range
    camarilla_h4 = camarilla_pp + 1.5 * camarilla_r    # H4 = PP + 1.5*range
    camarilla_l4 = camarilla_pp - 1.5 * camarilla_r    # L4 = PP - 1.5*range
    
    # Align Camarilla levels to 4h timeframe (yesterday's levels apply to today's 4h bars)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
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
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
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
        
        # Camarilla breakout conditions
        camarilla_breakout_long = close_price > camarilla_h3_aligned[i]
        camarilla_breakout_short = close_price < camarilla_l3_aligned[i]
        
        # Entry conditions
        enter_long = camarilla_breakout_long and vol_confirm and adx_filter
        enter_short = camarilla_breakout_short and vol_confirm and adx_filter
        
        # Exit conditions: price reverses to pivot point or touches opposite level (H4/L4)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops below pivot or touches L4 (mean reversion to PP)
            exit_long = close_price < camarilla_pp_aligned[i] or low_price <= camarilla_l4_aligned[i]
        elif position == -1:
            # Exit short if price rises above pivot or touches H4 (mean reversion to PP)
            exit_short = close_price > camarilla_pp_aligned[i] or high_price >= camarilla_h4_aligned[i]
        
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