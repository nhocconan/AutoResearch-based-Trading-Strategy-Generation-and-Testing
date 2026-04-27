#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime_Volume
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA50 trend filter and 1d chop regime filter (CHOP<38.2 = trending) and volume confirmation.
Only take long when price breaks above R1 and 4h trend up and 1d trending and volume spike.
Only take short when price breaks below S1 and 4h trend down and 1d trending and volume spike.
Designed for 15-35 trades/year on 1h to minimize fee drag while maintaining edge in both bull and bear markets.
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
    
    # Calculate Camarilla levels for 1h (based on previous bar's range)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # We need previous bar's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # fill first value
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12
    s1 = prev_close - 1.1 * camarilla_range / 12
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d Chopiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range(14))) / log10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
                                  np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))))
    tr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    max_high_14_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_14_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    range_14_1d = max_high_14_1d - min_low_14_1d
    chop_1d = 100 * np.log10(atr_14_1d * 14 / range_14_1d) / np.log10(14)
    chop_1d = np.where(range_14_1d > 0, chop_1d, 50)  # avoid division by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # already datetime64[ms], .hour works
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20  # 20% position size
    
    # Warmup: need enough for Camarilla (need prev bar), EMA50, Chop, volume average
    start_idx = max(1, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_50_4h_aligned[i]
        chop_val = chop_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        long_breakout = close_val > r1[i]
        short_breakout = close_val < s1[i]
        
        # Trend and regime filters
        trend_up = close_val > ema_trend
        trend_down = close_val < ema_trend
        trending_regime = chop_val < 38.2  # CHOP < 38.2 = trending
        
        if position == 0:
            # Flat - look for entry
            if long_breakout and trend_up and trending_regime and vol_spike:
                signals[i] = size
                position = 1
            elif short_breakout and trend_down and trending_regime and vol_spike:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price breaks below S1 or trend turns down or regime changes to ranging
            if close_val < s1[i] or not trend_up or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above R1 or trend turns up or regime changes to ranging
            if close_val > r1[i] or not trend_down or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime_Volume"
timeframe = "1h"
leverage = 1.0