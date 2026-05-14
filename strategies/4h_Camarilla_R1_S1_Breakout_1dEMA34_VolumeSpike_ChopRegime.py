#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_ChopRegime
Hypothesis: Camarilla R1/S1 breakout with volume confirmation and 1d EMA34 trend filter,
plus choppiness regime to avoid whipsaws in ranging markets. Long when price breaks above R1
with volume > 2.0x median volume, close > 1d EMA34, and CHOP > 61.8 (ranging = mean reversion
fails, so we only take breakouts in trending/low-chop). Short when price breaks below S1 with
volume confirmation, close < 1d EMA34, and CHOP > 61.8. Uses discrete sizing (0.25) to minimize
fee drag. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 4h (based on previous bar's range)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    
    # Volume confirmation: volume > 2.0x 20-period median (more robust than mean)
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (vol_median * 2.0)
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Choppiness Index regime filter (14-period)
    # CHOP > 61.8 = ranging (avoid breakout trades), CHOP < 38.2 = trending
    # We want LOW choppiness for breakout strategies (trending market)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    chop = np.where(atr14 > 0, 100 * np.log10((max_high - min_low) / atr14) / np.log10(14), 50)
    chop[np.isnan(chop)] = 50
    
    # Regime filter: only take breakouts when market is not too choppy (CHOP <= 61.8)
    regime_filter = chop <= 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period for volume median, 14 for chop)
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(regime_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R1 with volume confirmation, 1d uptrend, and favorable regime
        long_condition = (close[i] > r1[i]) and volume_confirm[i] and (close[i] > ema_34_1d_aligned[i]) and regime_filter[i]
        # Short logic: break below S1 with volume confirmation, 1d downtrend, and favorable regime
        short_condition = (close[i] < s1[i]) and volume_confirm[i] and (close[i] < ema_34_1d_aligned[i]) and regime_filter[i]
        
        # Exit logic: opposite Camarilla level touch or trend reversal
        exit_long = (close[i] < s1[i]) or (close[i] < ema_34_1d_aligned[i])
        exit_short = (close[i] > r1[i]) or (close[i] > ema_34_1d_aligned[i])
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0