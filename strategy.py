#!/usr/bin/env python3
"""
4h Bollinger Band Squeeze + Volume Spike + ADX Trend Filter
Long: BB squeeze (BBW < 20th percentile) + volume > 1.5x avg volume + ADX > 25 + close > upper BB
Short: BB squeeze + volume > 1.5x avg volume + ADX > 25 + close < lower BB
Exit: Opposite BB touch or BBW > 50th percentile (squeeze end)
Uses Bollinger Bands (20,2) for squeeze detection, volume confirmation, and ADX for trend strength.
Designed to capture breakouts from low volatility periods in both bull and bear markets.
Target: 80-160 total trades over 4 years (20-40/year)
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
    
    # Bollinger Bands (20,2)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = close_s.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    # Bollinger Band Width
    bbw = (upper - lower) / basis
    
    # Percentile rank of BBW (20-period lookback)
    bbw_series = pd.Series(bbw)
    bbw_percentile = bbw_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    
    # Average volume (50-period)
    vol_s = pd.Series(volume)
    avg_volume = vol_s.rolling(window=50, min_periods=20).mean()
    volume_spike = volume > (1.5 * avg_volume)
    
    # ADX (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14 + 1e-10)
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for additional trend filter (optional)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend bias
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    
    start_idx = 50  # need BB, ADX calculations
    
    for i in range(start_idx, n):
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(bbw_percentile[i]) or np.isnan(adx[i]) or
            np.isnan(volume_spike[i]) or np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: squeeze + volume spike + strong trend + close above upper BB + bullish bias
        long_condition = (bbw_percentile[i] < 20 and  # squeeze (low volatility)
                         volume_spike[i] and          # volume confirmation
                         adx[i] > 25 and              # strong trend
                         close[i] > upper[i] and      # breakout above upper band
                         close[i] > ema34_1d_aligned[i])  # bullish bias from 1d EMA
        
        # Short conditions: squeeze + volume spike + strong trend + close below lower BB + bearish bias
        short_condition = (bbw_percentile[i] < 20 and  # squeeze (low volatility)
                          volume_spike[i] and          # volume confirmation
                          adx[i] > 25 and              # strong trend
                          close[i] < lower[i] and      # breakdown below lower band
                          close[i] < ema34_1d_aligned[i])  # bearish bias from 1d EMA
        
        if long_condition:
            signals[i] = 0.25
        elif short_condition:
            signals[i] = -0.25
        else:
            # Exit conditions: squeeze ends (BBW > 50th percentile) or opposite BB touch
            if (bbw_percentile[i] > 50 or
                (signals[i-1] > 0 and close[i] < lower[i]) or
                (signals[i-1] < 0 and close[i] > upper[i])):
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]  # maintain position
    
    return signals

name = "4h_BB_Squeeze_Volume_ADX"
timeframe = "4h"
leverage = 1.0