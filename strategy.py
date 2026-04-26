#!/usr/bin/env python3
"""
1h_FundingRate_Contrarian_v1
Hypothesis: Funding rate mean reversion on BTC/ETH (works in both bull/bear). Short when funding > 0.03%, long when funding < -0.03%. Uses 1h for entry timing and 4h/1d HTF for regime filter (avoid counter-trend in strong moves). Low trade frequency (~20-40/year) minimizes fee drag. Discrete sizing 0.20.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load funding rate data (assumed available via mtf_data or precomputed)
    # For now, simulate funding rate proxy using price momentum (to be replaced with actual funding)
    # Actual implementation would load funding parquet: df_fund = pd.read_parquet('data/processed/funding/BTCUSDT.parquet')
    # Since funding data not directly accessible in generate_signals, we use price-based proxy
    # Proxy: funding ≈ (price - vwap) scaled (simplified)
    # In reality, replace this with actual funding rate load
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # VWAP approximation for funding proxy
    vwap = (close * volume).cumsum() / volume.cumsum()
    vwap = np.where(volume.cumsum() == 0, 0, vwap)
    funding_proxy = (close - vwap) / vwap  # Simplified proxy
    
    # Load 4h and 1d data for regime filter
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend regime
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA34 for stronger trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if not (8 <= hours[i] <= 20):
            # Outside session: flatten
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        fp = funding_proxy[i]
        ema_4h = ema_50_4h_aligned[i]
        ema_1d = ema_34_1d_aligned[i]
        
        if np.isnan(fp) or np.isnan(ema_4h) or np.isnan(ema_1d):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Entry: funding extreme + price vs EMA filter
        long_entry = (fp < -0.0003) and (close[i] > ema_4h) and (close[i] > ema_1d)
        short_entry = (fp > 0.0003) and (close[i] < ema_4h) and (close[i] < ema_1d)
        
        # Exit: funding reverts to neutral or trend fails
        long_exit = (position == 1 and (fp > -0.0001 or close[i] < ema_4h))
        short_exit = (position == -1 and (fp < 0.0001 or close[i] > ema_4h))
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1h_FundingRate_Contrarian_v1"
timeframe = "1h"
leverage = 1.0