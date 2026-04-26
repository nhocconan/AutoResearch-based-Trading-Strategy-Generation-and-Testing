#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dRegime_VolumeSpike
Hypothesis: On 1h timeframe, enter long when price breaks above daily Camarilla R1 AND 4h trend is up (close > EMA50) AND 1d chop regime is ranging (CHOP > 61.8) for mean reversion edge, with volume confirmation. Enter short on breakdown below S1 with 4h downtrend and ranging regime. Uses 1d Choppiness Index to filter for ranging markets where Camarilla mean reversion works best, reducing false breakouts in strong trends. Targets 15-30 trades/year by requiring confluence of 4h trend, 1d regime, and volume spike.
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
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + 1.1 * camarilla_range / 12
    s1 = prev_close_1d - 1.1 * camarilla_range / 12
    mid = (r1 + s1) / 2  # Camarilla midpoint
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    mid_aligned = align_htf_to_ltf(prices, df_1d, mid)
    
    # Calculate 1d Choppiness Index (CHOP) for regime detection
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    # Where n = 14 periods
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.maximum(max_high14 - min_low14, 1e-10)
    chop = 100 * np.log10(sum_atr14 / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 2.0x 24-period average on 1h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50, volume MA, and CHOP warmup
    start_idx = max(50, 24, 30)  # EMA50(4h) + volume MA24 + CHOP warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i]) or not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        ranging_regime = chop_aligned[i] > 61.8
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        
        # 4h trend filter
        trend_uptrend = close[i] > ema_50_4h_aligned[i]
        trend_downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume spike + 4h uptrend + ranging regime
            long_signal = breakout_up and volume_spike[i] and trend_uptrend and ranging_regime
            
            # Short: breakout below S1 + volume spike + 4h downtrend + ranging regime
            short_signal = breakout_down and volume_spike[i] and trend_downtrend and ranging_regime
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: trend change to downtrend OR price retracing to Camarilla midpoint OR regime shifts to trending
            if (not trend_uptrend) or (close[i] < mid_aligned[i]) or (chop_aligned[i] <= 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: trend change to uptrend OR price retracing to Camarilla midpoint OR regime shifts to trending
            if (not trend_downtrend) or (close[i] > mid_aligned[i]) or (chop_aligned[i] <= 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dRegime_VolumeSpike"
timeframe = "1h"
leverage = 1.0