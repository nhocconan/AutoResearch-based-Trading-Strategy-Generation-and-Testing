#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_FundingZ_v2
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d EMA trend filter and funding rate mean reversion. 
Only trade breakouts aligned with 1d EMA34 trend and extreme funding rate Z-score. 
Funding rate provides BTC/ETH edge in both bull and bear markets by capturing sentiment extremes.
Uses tighter entry conditions to reduce trade count and avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    R1 = PP + (high_1d - low_1d) * 1.0 / 4.0
    S1 = PP - (high_1d - low_1d) * 1.0 / 4.0
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA34 trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Funding rate mean reversion (Z-score of funding)
    # Load funding rate data for the symbol
    symbol = getattr(prices, 'name', 'BTCUSDT')  # fallback if name not set
    try:
        funding_path = f"/mnt/shared/funding/{symbol}.parquet"
        df_funding = pd.read_parquet(funding_path)
        # Align funding rate to 4h timeframe (funding is every 8h)
        funding_rate = df_funding['fundingRate'].values
        funding_times = df_funding['timestamp'].values
        # Create funding series aligned to prices index
        funding_series = pd.Series(index=prices.index, dtype=float)
        for ts, rate in zip(funding_times, funding_rate):
            mask = prices['open_time'] == ts
            if mask.any():
                funding_series.loc[mask] = rate
        funding_values = funding_series.values
        # Calculate 30-day Z-score
        funding_mean = pd.Series(funding_values).rolling(window=90, min_periods=30).mean().values  # 90 * 4h = 30 days
        funding_std = pd.Series(funding_values).rolling(window=90, min_periods=30).std().values
        funding_z = (funding_values - funding_mean) / (funding_std + 1e-8)
    except:
        # If funding data not available, use neutral Z-score
        funding_z = np.zeros(n)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(100, 34, 20, 90)  # EMA, volume, funding
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(funding_z[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout with volume confirmation and trend/funding filters
            # Long: price > R1, above 1d EMA34, funding extremely negative (oversold)
            # Short: price < S1, below 1d EMA34, funding extremely positive (overbought)
            long_entry = (close_val > R1_aligned[i]) and (close_val > ema34_1d_aligned[i]) and volume_spike[i] and (funding_z[i] < -2.0)
            short_entry = (close_val < S1_aligned[i]) and (close_val < ema34_1d_aligned[i]) and volume_spike[i] and (funding_z[i] > 2.0)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on S1 retracement or funding normalization
            if close_val < S1_aligned[i] or funding_z[i] > -0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R1 retracement or funding normalization
            if close_val > R1_aligned[i] or funding_z[i] < 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_FundingZ_v2"
timeframe = "4h"
leverage = 1.0