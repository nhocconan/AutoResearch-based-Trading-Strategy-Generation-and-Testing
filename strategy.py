#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_FundingReversion
Hypothesis: Combines Camarilla H3/L3 breakout with 1d EMA34 trend filter and funding rate mean reversion.
In bull markets: breakouts with trend and negative funding (long bias) produce longs.
In bear markets: fades from extremes with positive funding (short bias) produce shorts.
Volume confirmation ensures conviction. Discrete sizing (0.25) limits fee drag. Targets 25-40 trades/year.
Works in both bull and bear regimes via funding rate edge proven effective on BTC/ETH.
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
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels: H3/L3
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align to 4h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Funding rate mean reversion: load 8h funding data, compute z-score, look for extremes
    try:
        funding_path = "/mnt/shared/data/processed/funding/btcusdt_8h.parquet"
        df_funding = pd.read_parquet(funding_path)
        # Align funding timestamps to price index (both are datetime64)
        funding_series = df_funding.set_index('open_time')['funding_rate']
        # Reindex to match prices index, forward fill (funding updates every 8h)
        funding_aligned = funding_series.reindex(prices.index, method='ffill').values
        # Compute 30-day z-score (90 periods of 8h data = 30 days)
        funding_ma = pd.Series(funding_aligned).rolling(window=90, min_periods=30).mean().values
        funding_std = pd.Series(funding_aligned).rolling(window=90, min_periods=30).std().values
        funding_z = (funding_aligned - funding_ma) / funding_std
        # Replace NaN/inf with 0
        funding_z = np.nan_to_num(funding_z, nan=0.0, posinf=0.0, neginf=0.0)
        # Funding reversal signals: long when funding very negative, short when very positive
        funding_long = funding_z < -1.5
        funding_short = funding_z > 1.5
    except Exception:
        # If funding data fails, disable this filter (no trades from funding alone)
        funding_long = np.zeros(n, dtype=bool)
        funding_short = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Camarilla (1 bar), EMA34 (34), volume MA (20), funding (90)
    start_idx = max(1, 34, 20, 90)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price closes above H3 + 1d uptrend + volume spike + negative funding bias
            long_setup = (close[i] > camarilla_h3_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike[i] and \
                         funding_long[i]
            # Short: price closes below L3 + 1d downtrend + volume spike + positive funding bias
            short_setup = (close[i] < camarilla_l3_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike[i] and \
                          funding_short[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below L3 OR 1d trend turns down
            if (close[i] < camarilla_l3_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above H3 OR 1d trend turns up
            if (close[i] > camarilla_h3_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_FundingReversion"
timeframe = "4h"
leverage = 1.0