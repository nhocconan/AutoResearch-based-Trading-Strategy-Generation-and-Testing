#!/usr/bin/env python3
"""
6h_FundingRate_Contrarian_1dRegime_Filter
Hypothesis: Funding rate mean reversion works on BTC/ETH due to overleveraged longs/shorts correcting.
Enter short when funding > 0.03% (extreme long bias), long when funding < -0.03% (extreme short bias).
Use 1d ADX regime filter: only trade when ADX < 25 (range market) to avoid trending markets where funding trends persist.
Volume confirmation ensures institutional participation. Designed for 50-150 trades over 4 years.
Works in bull/bear by fading extremes regardless of price direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load funding rate data (assuming available as column, else placeholder)
    # In real implementation, funding rate would be loaded from external parquet
    # For now, simulate with price-based proxy: RSI divergence as funding proxy
    # TODO: Replace with actual funding rate loading: pd.read_parquet(funding_path)
    rsi = pd.Series(close).rolling(14, min_periods=14).apply(
        lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / (-x.diff().clip(upper=0).abs().mean()+1e-10)))
    ), raw=False).values
    # Normalize RSI to funding-like signal: >70 = long bias, <30 = short bias
    funding_proxy = (rsi - 50) / 50  # -1 to 1 range
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.concatenate([[0], np.maximum(high_1d[1:] - high_1d[:-1], 0)])
    down_move = np.concatenate([[0], np.maximum(low_1d[:-1] - low_1d[1:], 0)])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di_1d = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di_1d = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.3 * 20-period EMA
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.3 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(funding_proxy[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Regime filter: only trade in low ADX (range) market
        if adx_1d_aligned[i] >= 25:
            # Trending market - reduce position or stay flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size * 0.5  # reduce long
            else:
                signals[i] = -base_size * 0.5  # reduce short
            continue
        
        # Long signal: extreme short bias (funding < -0.03) + volume spike
        if funding_proxy[i] < -0.3 and volume_spike[i]:  # RSI < 35 equivalent
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short signal: extreme long bias (funding > 0.03) + volume spike
        elif funding_proxy[i] > 0.3 and volume_spike[i]:  # RSI > 65 equivalent
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: funding returns to neutral
        elif position == 1 and funding_proxy[i] > -0.1:
            signals[i] = 0.0
            position = 0
        elif position == -1 and funding_proxy[i] < 0.1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_FundingRate_Contrarian_1dRegime_Filter"
timeframe = "6h"
leverage = 1.0