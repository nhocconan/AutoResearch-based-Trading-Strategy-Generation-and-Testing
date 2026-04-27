#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Bands with 1w ADX trend filter and volume confirmation.
# Bollinger Bands (BB) identify volatility-based support/resistance.
# BB squeeze + expansion signals potential breakout/breakdown.
# ADX > 25 indicates trending market, ADX < 20 indicates ranging market.
# Strategy: In ranging markets (ADX < 20), buy at lower BB, sell at upper BB (mean reversion).
# In trending markets (ADX > 25), break above upper BB = long, break below lower BB = short.
# Volume spike confirms institutional participation.
# Designed for ~15-25 trades/year per symbol (60-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands calculation (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma + (bb_std * std)
    lower_bb = sma - (bb_std * std)
    
    # ADX calculation (14-period) on weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]  # first period
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = np.diff(low_1w, prepend=low_1w[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_period = 14
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/tr_period, adjust=False).mean().values
    plus_di_1w = 100 * (pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values / atr_1w)
    minus_di_1w = 100 * (pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values / atr_1w)
    
    # DX and ADX
    dx_denom = plus_di_1w + minus_di_1w
    dx_denom = np.where(dx_denom == 0, 1e-10, dx_denom)
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / dx_denom
    adx_1w = pd.Series(dx_1w).ewm(alpha=1/tr_period, adjust=False).mean().values
    
    # Align ADX to daily timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX < 20 = ranging market (mean reversion)
        # ADX > 25 = trending market (breakout)
        if adx_1w_aligned[i] < 20:  # Ranging market
            # Buy near lower BB, sell near upper BB
            if close[i] <= lower_bb[i] * 1.005 and close[i] >= lower_bb[i] * 0.995:  # Near lower BB
                if volume_filter[i]:  # Volume confirmation
                    signals[i] = 0.25
                    position = 1
            elif close[i] >= upper_bb[i] * 0.995 and close[i] <= upper_bb[i] * 1.005:  # Near upper BB
                if volume_filter[i]:  # Volume confirmation
                    signals[i] = -0.25
                    position = -1
        elif adx_1w_aligned[i] > 25:  # Trending market
            # Break above upper BB = long, break below lower BB = short
            if close[i] > upper_bb[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] < lower_bb[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                # Hold position in trend
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # Transition zone (20 <= ADX <= 25) - hold or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_BollingerBands_1wADX_VolumeFilter"
timeframe = "1d"
leverage = 1.0