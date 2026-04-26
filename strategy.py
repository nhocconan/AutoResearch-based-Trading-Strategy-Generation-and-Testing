#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Trade 12h Camarilla R3/S3 breakouts with 1d EMA34 trend filter, volume confirmation, and chop regime filter.
Targets 75-175 total trades over 4 years (19-44/year) on 12h timeframe.
Uses Camarilla R3/S3 levels (wider than R1/S1) for fewer but higher-quality breakouts.
1d EMA34 ensures trading with dominant long-term trend.
Volume confirmation adds conviction to breakouts.
Choppiness regime filter (CHOP < 38.2) ensures trending market where breakouts work.
Designed for BTC/ETH with SOL as secondary - works in bull (breakouts with trend) and bear (mean reversion at extremes with trend filter).
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr1 = np.concatenate([[0], tr1])  # align length
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 4.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    
    # Align Camarilla levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma)
    
    # Choppiness regime filter: CHOP(14) < 38.2 = trending market (good for breakouts)
    hl_range = pd.Series(high - low).rolling(window=14, min_periods=14).sum()
    true_range = pd.Series(tr1).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(hl_range / true_range) / np.log10(14)
    chop_regime = chop < 38.2  # trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of 1d EMA(34), volume MA(20), ATR(14), CHOP(14)
    start_idx = max(34, 20, 14, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        trend_1d_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_1d_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        regime_ok = chop_regime[i]  # trending market regime
        
        if position == 0:
            # Long: price breaks above R3 AND volume confirm AND 1d trend up AND trending regime
            long_signal = (close_val > r3_aligned[i]) and vol_conf and trend_1d_up and regime_ok
            
            # Short: price breaks below S3 AND volume confirm AND 1d trend down AND trending regime
            short_signal = (close_val < s3_aligned[i]) and vol_conf and trend_1d_down and regime_ok
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # ATR trailing stop: exit if price drops 2.0 * ATR from highest since entry
            if close_val < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: trend flips down or regime changes to ranging
            elif not trend_1d_up or not regime_ok:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # ATR trailing stop: exit if price rises 2.0 * ATR from lowest since entry
            if close_val > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: trend flips up or regime changes to ranging
            elif not trend_1d_down or not regime_ok:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0