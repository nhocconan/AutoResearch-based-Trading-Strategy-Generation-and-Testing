#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_VolumeSpike_ChopRegime_ATRStop
Hypothesis: On 1d timeframe, Donchian(20) breakouts with volume confirmation and choppiness regime filter capture sustainable trends while avoiding false breakouts in ranging markets. Long when price breaks above upper Donchian channel with volume spike and chop > 61.8 (trending regime); Short when price breaks below lower Donchian channel with volume spike and chop > 61.8. Uses ATR-based stoploss and discrete sizing (±0.25) to minimize fee drag. Designed for 7-25 trades/year with strong BTC/ETH edge in both bull and bear markets.
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for higher-timeframe trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Average True Range for volatility and stoploss
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume spike detection (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2x average
    
    # Choppiness Index regime filter (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log10(highest_high - lowest_low))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We want trending regime: CHOP < 38.2
    atr_14 = tr.rolling(window=14, min_periods=14).sum().values
    hh_14 = high_series.rolling(window=14, min_periods=14).max().values
    ll_14 = low_series.rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_14 / ((hh_14 - ll_14) * np.log10(14))) / np.log10(14)
    # Handle division by zero and invalid values
    chop_raw = np.where((hh_14 - ll_14) > 0, chop_raw, 50.0)  # Neutral when no range
    chop_raw = np.where(np.isnan(chop_raw), 50.0, chop_raw)
    trending_regime = chop_raw < 38.2  # Trending when chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    
    # Warmup: max of Donchian (20), ATR (14*2=28), volume MA (20), chop (14), 1w EMA50 alignment
    start_idx = max(20, 28, 20, 14) + 4  # +4 to ensure 1w bar completion (1d -> 1w: ~7 bars per week, conservative)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr[i]) or np.isnan(volume_spike[i]) or
            np.isnan(trending_regime[i]) or np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        trending = trending_regime[i]
        ema_50_val = ema_50_1w_aligned[i]
        
        # Breakout conditions
        long_breakout = close_val > donchian_upper[i]
        short_breakout = close_val < donchian_lower[i]
        
        # Entry conditions: breakout + volume spike + trending regime + higher-timeframe trend
        long_entry = long_breakout and vol_spike and trending and (close_val > ema_50_val)
        short_entry = short_breakout and vol_spike and trending and (close_val < ema_50_val)
        
        # Stoploss conditions: ATR-based trailing stop
        long_stop = False
        short_stop = False
        
        if position == 1:  # Long position
            # Trailing stop: highest high since entry minus 2.5 * ATR
            # Simplified: stop if price drops below entry price - 2.5 * ATR (close-based)
            if entry_price > 0:
                long_stop = close_val < (entry_price - 2.5 * atr_val)
        elif position == -1:  # Short position
            # Trailing stop: lowest low since entry plus 2.5 * ATR
            # Simplified: stop if price rises above entry price + 2.5 * ATR (close-based)
            if entry_price > 0:
                short_stop = close_val > (entry_price + 2.5 * atr_val)
        
        # Exit conditions: stoploss or breakout in opposite direction
        long_exit = long_stop or short_breakout
        short_exit = short_stop or long_breakout
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_Donchian20_Breakout_VolumeSpike_ChopRegime_ATRStop"
timeframe = "1d"
leverage = 1.0