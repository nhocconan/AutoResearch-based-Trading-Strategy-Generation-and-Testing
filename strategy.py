#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_Regime
Hypothesis: Camarilla R1/S1 breakout on 12h with 1w EMA50 trend filter, volume spike, and chop regime filter.
Designed for 12-37 trades/year on BTC/ETH/SOL. Uses 1w trend for major market direction, volume confirmation for institutional participation,
and chop regime filter to avoid whipsaws. Should work in bull (breakouts with volume+trend) and bear (fade false breakouts, regime filter prevents wrong-way trades).
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
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
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike: current volume > 2.0 * 20-period average (moderate)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period on 12h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop regime filter: Bollinger Band Width percentile on 12h
    # Low BW (< 30th percentile) = ranging (avoid), High BW (> 70th) = trending (favor)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    bb_width = (upper_band - lower_band) / sma
    # Percentile of bb_width over last 50 periods
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: np.percentile(x, 70) if len(x) >= 20 else np.nan, raw=False
    ).values
    chop_filter = bb_width > bb_width_percentile  # True when BB width > 70th percentile = trending
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(100, 50, 20, 14, 50)  # EMA, volume avg, ATR, BB percentile
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        ema_trend = ema_1w_aligned[i]
        chop_ok = chop_filter[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout in direction of 1w trend with volume confirmation AND chop filter
            # Long: price above 1w EMA50 AND break above R1 + volume spike + trending regime
            long_entry = (close_val > ema_trend) and (close_val > R1_aligned[i]) and volume_spike[i] and chop_ok
            # Short: price below 1w EMA50 AND break below S1 + volume spike + trending regime
            short_entry = (close_val < ema_trend) and (close_val < S1_aligned[i]) and volume_spike[i] and chop_ok
            
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
            # Long - exit on S1 retracement or ATR stoploss
            exit_condition = (close_val < S1_aligned[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R1 retracement or ATR stoploss
            exit_condition = (close_val > R1_aligned[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0