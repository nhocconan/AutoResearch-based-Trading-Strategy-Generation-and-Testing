#!/usr/bin/env python3
"""
1h_VolumeSpike_Reversal_4hTrend_1dChopFilter_v1
Hypothesis: Trade mean-reversion volume spikes on 1h with 4h trend filter and 1d chop regime filter.
In ranging markets (CHOP > 61.8): fade volume spikes (long on down spikes, short on up spikes).
In trending markets (CHOP < 38.2): trade with 4h trend (long on up spikes in uptrend, short on down spikes in downtrend).
Volume spike defined as current volume > 2.0 * 20-period average volume.
Exit on opposite volume spike or trend/chop regime change.
Position size: 0.20 to limit drawdown and enable multiple concurrent positions.
Target: 20-40 trades/year to stay under 1h hard max of 60-150/year.
Works in bull (mean reversion in ranges, trend continuation in trends) and bear (same logic) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for HTF trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Chopiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d bars
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest High and Lowest Low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: CHOP = 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop = np.where(
        (range_14 > 0) & (~np.isnan(range_14)) & (~np.isnan(sum_tr_14)) & (~np.isnan(atr_14)),
        100 * np.log10(sum_tr_14 / range_14) / np.log10(14),
        np.nan
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Determine 1h price direction of volume spike
    price_up_spike = volume_spike & (close > open_time)  # close > open approximates bullish candle
    price_down_spike = volume_spike & (close < open_time)  # close < open approximates bearish candle
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), CHOP (14), volume MA (20)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine market regime from 1d chop
        chop_value = chop_aligned[i]
        is_ranging = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        # Determine 4h HTF trend
        htf_4h_bullish = close[i] > ema_34_4h_aligned[i]
        htf_4h_bearish = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Entry logic based on regime
            long_entry = False
            short_entry = False
            
            if is_ranging:
                # In ranging markets: fade volume spikes (mean reversion)
                long_entry = price_down_spike[i]  # short spike -> long
                short_entry = price_up_spike[i]   # long spike -> short
            elif is_trending:
                # In trending markets: trade with 4h trend
                long_entry = price_up_spike[i] & htf_4h_bullish   # up spike in uptrend
                short_entry = price_down_spike[i] & htf_4h_bearish # down spike in downtrend
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: opposite volume spike OR regime/trend change
            if (price_up_spike[i]) or (not is_ranging and not (is_trending and htf_4h_bullish)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: opposite volume spike OR regime/trend change
            if (price_down_spike[i]) or (not is_ranging and not (is_trending and htf_4h_bearish)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_VolumeSpike_Reversal_4hTrend_1dChopFilter_v1"
timeframe = "1h"
leverage = 1.0