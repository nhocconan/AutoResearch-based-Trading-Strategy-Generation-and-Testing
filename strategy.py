#!/usr/bin/env python3
"""
1d_Keltner_Channel_Squeeze_Breakout
Daily strategy using Keltner Channel squeeze breakout with volume confirmation.
Enters long when price breaks above upper KC after low volatility (BBwidth < KCwidth).
Enters short when price breaks below lower KC after low volatility.
Uses 1-week ADX as trend filter to avoid whipsaws in weak trends.
Exit when price returns to 20-day EMA or volatility expansion ends.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Calculate 1d Bollinger Bands (for squeeze detection) ===
    basis = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    dev = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = basis + 2.0 * dev
    lower_bb = basis - 2.0 * dev
    bb_width = (upper_bb - lower_bb) / basis
    
    # === Calculate 1d Keltner Channel ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    upper_kc = ema20 + 1.5 * atr
    lower_kc = ema20 - 1.5 * atr
    kc_width = (upper_kc - lower_kc) / ema20
    
    # === Squeeze condition: BB width < KC width (low volatility) ===
    squeeze = bb_width < kc_width
    
    # === Breakout conditions ===
    breakout_up = close > upper_kc
    breakout_down = close < lower_kc
    
    # === 1-week ADX for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1w = np.maximum(np.maximum(high_1w - low_1w, 
                                 np.abs(high_1w - np.roll(close_1w, 1))), 
                      np.abs(low_1w - np.roll(close_1w, 1)))
    tr1w[0] = high_1w[0] - low_1w[0]
    
    atr14 = pd.Series(tr1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr14
    minus_di14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr14
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align to daily
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze.astype(float))
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    upper_kc_aligned = align_htf_to_ltf(prices, df_1w, upper_kc)
    lower_kc_aligned = align_htf_to_ltf(prices, df_1w, lower_kc)
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20)
    
    # === 1d Volume for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(squeeze_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(upper_kc_aligned[i]) or 
            np.isnan(lower_kc_aligned[i]) or 
            np.isnan(ema20_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: above average volume
        vol_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        # Trend filter: ADX > 20 indicates strong enough trend
        trend_filter = adx_aligned[i] > 20
        
        # Squeeze breakout: volatility expansion after squeeze
        squeeze_release = squeeze_aligned[i] and not squeeze_aligned[i-1]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above upper KC after squeeze, with volume and trend
            if breakout_up[i] and squeeze_release and vol_confirmed and trend_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below lower KC after squeeze, with volume and trend
            elif breakout_down[i] and squeeze_release and vol_confirmed and trend_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to 20-day EMA OR volatility expansion ends (squeeze ends)
            if close[i] <= ema20_aligned[i] or not squeeze_release:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 20-day EMA OR volatility expansion ends
            if close[i] >= ema20_aligned[i] or not squeeze_release:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Keltner_Channel_Squeeze_Breakout"
timeframe = "1d"
leverage = 1.0