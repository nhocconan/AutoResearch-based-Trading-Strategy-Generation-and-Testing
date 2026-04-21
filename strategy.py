#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime_v1
Hypothesis: Camarilla R1/S1 breakouts on 1h filtered by 4h EMA50 trend and 1d chop regime (range: mean revert, trend: follow). 
Uses Bollinger Band Width percentile on 1d to detect regime: CHOP > 60 = range (fade R1/S1), CHOP < 40 = trend (breakout). 
Session filter 08-20 UTC reduces noise. Position size 0.20 for balanced risk. 
Target: 20-40 trades/year per symbol for low fee drag and strong test generalization in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1h OHLC ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Previous day's Camarilla levels (from 1d) ===
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    rng = d_high - d_low
    r1 = d_close + 0.275 * rng
    s1 = d_close - 0.275 * rng
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 4h EMA50 for trend filter ===
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d Bollinger Band Width for regime detection ===
    bb_mid = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean()
    bb_std = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    bb_width_percentile = bb_width.rolling(window=100, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # open_time is datetime64[ms], index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(ema_50_4h_aligned[i]) or np.isnan(bb_width_percentile_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        price = close[i]
        
        if position == 0 and in_session:
            # Regime-based logic
            chop = bb_width_percentile_aligned[i]
            is_range = chop > 60.0  # high chop = range -> mean revert
            is_trend = chop < 40.0  # low chop = trend -> follow breakout
            
            # Volume filter: current volume > 1.5x 20-period average
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_filter = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            if is_range:
                # In range: fade R1/S1 (mean reversion)
                long_condition = price < s1_aligned[i] and vol_filter
                short_condition = price > r1_aligned[i] and vol_filter
            elif is_trend:
                # In trend: breakout R1/S1 (follow momentum)
                long_condition = price > r1_aligned[i] and vol_filter
                short_condition = price < s1_aligned[i] and vol_filter
            else:
                # Neutral chop: no trade
                long_condition = False
                short_condition = False
            
            # Trend filter: 4h EMA50 alignment
            if long_condition:
                long_trend = price > ema_50_4h_aligned[i] if is_trend else True  # in range, trend less critical
                if long_trend:
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
            elif short_condition:
                short_trend = price < ema_50_4h_aligned[i] if is_trend else True  # in range, trend less critical
                if short_trend:
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Exit conditions
            if is_range:
                # In range: exit at opposite S1/R1 or midpoint
                exit_long = price > s1_aligned[i] + 0.5 * (r1_aligned[i] - s1_aligned[i])
            else:
                # In trend: exit on close below EMA50 or S1 breach
                exit_long = price < ema_50_4h_aligned[i] or price < s1_aligned[i]
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit conditions
            if is_range:
                # In range: exit at opposite S1/R1 or midpoint
                exit_short = price < r1_aligned[i] - 0.5 * (r1_aligned[i] - s1_aligned[i])
            else:
                # In trend: exit on close above EMA50 or R1 breach
                exit_short = price > ema_50_4h_aligned[i] or price > r1_aligned[i]
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime_v1"
timeframe = "1h"
leverage = 1.0