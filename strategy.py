#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_ChopRegime_ATRStop
Hypothesis: On 4h timeframe, Donchian(20) breakouts with volume confirmation and choppy regime filter (CHOP > 61.8) capture sustainable trends while avoiding false breakouts in ranging markets. Uses ATR-based trailing stop for risk control. Designed for 20-50 trades/year with discrete sizing (±0.25) to minimize fee drag and work in both bull/bear markets with BTC/ETH edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d True Range for Choppy Market Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Choppy Market Index: CHOP = 100 * log10(sum(ATR14) / (max(HH) - min(LL))) / log10(14)
    # We use a simplified version: CHOP = 100 * log10(rolling_sum(ATR14, 14) / (rolling_max(high,14) - rolling_min(low,14))) / log10(14)
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (hh_14 - ll_14 + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw, additional_delay_bars=0)
    
    # 4h Donchian Channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 4h ATR for stop loss and volume average
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Donchian (20), ATR (14), volume MA (20), 1d CHOP (14+14=28)
    start_idx = max(lookback, 14, 20) + 4  # +4 to ensure 1d bar completion (4h -> 1d: 6 bars per day)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        vol_val = volume[i]
        
        # Regime conditions: CHOP > 61.8 indicates choppy/ranging market (favor mean reversion, but we use it to avoid false breakouts)
        # Actually, CHOP > 61.8 = ranging, CHOP < 38.2 = trending
        # We want to avoid breakouts in ranging markets, so we require CHOP < 61.8 (not strongly ranging)
        not_strongly_ranging = chop_aligned[i] < 61.8
        
        # Volume confirmation: volume > 1.5 * 20-period average
        volume_spike = vol_val > 1.5 * vol_ma[i]
        
        # Entry conditions
        long_entry = (close_val > highest_high[i]) and not_strongly_ranging and volume_spike
        short_entry = (close_val < lowest_low[i]) and not_strongly_ranging and volume_spike
        
        # Exit conditions: ATR-based trailing stop or opposite Donchian break
        if position == 1:  # Long position
            # Trail stop: highest high since entry minus 2.5 * ATR
            # Simplified: exit if price drops below entry point - 2.5*ATR (we don't track entry price, so use Donchian lower band)
            long_exit = (close_val < lowest_low[i]) or (close_val < close[i-1] - 2.5 * atr[i]) if i > 0 else False
        elif position == -1:  # Short position
            # Trail stop: lowest low since entry plus 2.5 * ATR
            short_exit = (close_val > highest_high[i]) or (close_val > close[i-1] + 2.5 * atr[i]) if i > 0 else False
        else:
            long_exit = False
            short_exit = False
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_VolumeSpike_ChopRegime_ATRStop"
timeframe = "4h"
leverage = 1.0