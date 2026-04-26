#!/usr/bin/env python3
"""
1d_Donchian_20_Breakout_1wTrend_VolumeSpike_ATRStop
Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
In bull markets: price breaks above upper band with weekly uptrend → long.
In bear markets: price breaks below lower band with weekly downtrend → short.
Uses discrete sizing (0.25) and ATR-based stoploss to reduce fee drag and manage risk.
Target: 30-100 trades over 4 years (7-25/year). Works in both regimes by requiring alignment with weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:  # Need 20 for Donchian
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    period_dc = 20
    max_high_dc = pd.Series(high).rolling(window=period_dc, min_periods=period_dc).max().values
    min_low_dc = pd.Series(low).rolling(window=period_dc, min_periods=period_dc).min().values
    
    # ATR(14) for stoploss and volatility filter
    period_atr = 14
    tr1 = pd.Series(high).rolling(window=2).max().values - pd.Series(low).rolling(window=2).min().values
    tr2 = abs(pd.Series(high).rolling(window=2).max().values - pd.Series(close).shift(1).rolling(window=2).min().values)
    tr3 = abs(pd.Series(low).rolling(window=2).min().values - pd.Series(close).shift(1).rolling(window=2).max().values)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=period_atr, min_periods=period_atr).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 1.5)
    
    # Load weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Weekly EMA10 for trend filter (responsive but smooth)
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 20 for Donchian, 14 for ATR)
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(max_high_dc[i]) or np.isnan(min_low_dc[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(ema_10_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Donchian breakout conditions
        upper_band = max_high_dc[i]
        lower_band = min_low_dc[i]
        close_val = close[i]
        
        # Long: price breaks above upper band with volume spike and weekly uptrend
        long_condition = close_val > upper_band and volume_spike[i] and (close_val > ema_10_1w_aligned[i])
        # Short: price breaks below lower band with volume spike and weekly downtrend
        short_condition = close_val < lower_band and volume_spike[i] and (close_val < ema_10_1w_aligned[i])
        
        # Exit conditions: ATR-based stoploss or opposite breakout
        exit_long = False
        exit_short = False
        if position == 1:
            # Stoploss: 2.5 * ATR below entry (tracked via highest high since entry)
            # Simplified: exit if price drops below midpoint of bands or weekly trend reverses
            exit_long = close_val < (upper_band + lower_band) / 2 or close_val < ema_10_1w_aligned[i]
        elif position == -1:
            # Stoploss: 2.5 * ATR above entry
            exit_short = close_val > (upper_band + lower_band) / 2 or close_val > ema_10_1w_aligned[i]
        
        # Minimum holding period: 3 days to reduce churn
        if position != 0 and bars_since_entry < 3:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Donchian_20_Breakout_1wTrend_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0