#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeChopFilter
Hypothesis: On 4h timeframe, price breaking above/below 20-period Donchian channels in direction of 1d EMA50 trend with volume confirmation (>1.3x 20-period MA) and choppy market filter (Choppiness Index > 50) captures trend continuation with reduced false breakouts. Volume spike confirms institutional participation, chop filter avoids whipsaws in ranging markets. Discrete sizing (±0.25) and ATR trailing stop (2.5x) minimize fee drag. Designed for 20-50 trades/year with BTC/ETH edge in bull/bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d Choppiness Index (14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Choppiness Index: 100 * log10(sum(ATR14) / (n * log10(highest_high - lowest_low))) / log10(n)
    sum_atr14 = atr_14.rolling(window=14, min_periods=14).sum()
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop_denominator = 14 * np.log10(hh14 - ll14)
    chop_ratio = sum_atr14 / chop_denominator.replace(0, np.nan)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_values = chop.fillna(50).values  # neutral when undefined
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    donchian_high_values = donchian_high.values
    donchian_low_values = donchian_low.values
    
    # 4h ATR(20) for trailing stop
    tr1_4h = pd.Series(high).diff().abs()
    tr2_4h = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3_4h = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_4h = tr_4h.ewm(span=20, adjust=False, min_periods=20).mean()
    atr_4h_values = atr_4h.values
    
    # Volume spike filter: volume > 1.3 * 20-period MA on 4h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of Donchian (20), ATR (20), volume MA (20) + time for 1d alignment
    start_idx = max(20, 20, 20) + 4  # +4 to ensure 1d bar completion (4h -> 1d: 6 bars per day)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        donchian_high_val = donchian_high_values[i]
        donchian_low_val = donchian_low_values[i]
        ema_val = ema_50_1d_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(donchian_high_val) or np.isnan(donchian_low_val) or 
            np.isnan(ema_val) or np.isnan(chop_val) or np.isnan(atr_val) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: bullish when price > EMA50, bearish when price < EMA50
        trend_bullish = close_val > ema_val
        trend_bearish = close_val < ema_val
        
        # Chop filter: only trade in choppy/range markets (Chop > 50)
        chop_filter = chop_val > 50
        
        # Donchian breakout conditions: price breaks channel with trend alignment + volume spike + chop filter
        long_breakout = close_val > donchian_high_val
        short_breakout = close_val < donchian_low_val
        
        long_entry = trend_bullish and long_breakout and vol_spike and chop_filter
        short_entry = trend_bearish and short_breakout and vol_spike and chop_filter
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.5 * ATR
            stop_price = highest_since_long - 2.5 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.5 * ATR
            stop_price = lowest_since_short + 2.5 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeChopFilter"
timeframe = "4h"
leverage = 1.0