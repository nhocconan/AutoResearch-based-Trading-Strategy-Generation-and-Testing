#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_Regime_v2
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike (>2.0x 20-bar MA), and choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend) captures strong directional moves with controlled frequency. Uses discrete sizing (0.25) to minimize fee churn. Works in bull/bear by taking breakouts only in trend direction when market is trending (CHOP < 38.2).
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
    
    # Load 1d data ONCE before loop for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    # Choppiness regime filter: CHOP(14) < 38.2 = trending (good for breakouts)
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14.sum() / np.log(14) / (highest_high14 - lowest_low14)) if False else np.zeros(n)  # placeholder
    # Recalculate properly with vectorized approach
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range14 = highest_high14 - lowest_low14
    range14 = np.where(range14 == 0, 1e-10, range14)
    chop = 100 * np.log10(atr14 * np.sqrt(14) / range14) / np.log10(10)
    chop[np.isnan(chop)] = 50  # default to middle range when undefined
    chop_filter = chop < 38.2  # only trade when market is trending
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of EMA34 (34), volume MA (20), CHOP (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        trend_val = ema34_1d_aligned[i]
        vol_conf = volume_confirm[i]
        chop_ok = chop_filter[i]
        
        # Skip if any data not ready
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(trend_val) or np.isnan(chop_ok)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > EMA34 = uptrend, price < EMA34 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Entry conditions: Camarilla breakout in direction of trend + volume + chop filter
        long_condition = (close_val > r1_val) and is_uptrend and vol_conf and chop_ok
        short_condition = (close_val < s1_val) and is_downtrend and vol_conf and chop_ok
        
        # Exit conditions: opposite Camarilla level touch or trend reversal or chop regime change to range
        long_exit = (position == 1 and (close_val < s1_val or not is_uptrend or not chop_ok))
        short_exit = (position == -1 and (close_val > r1_val or not is_downtrend or not chop_ok))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_Regime_v2"
timeframe = "4h"
leverage = 1.0