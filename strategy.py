#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: On 12h timeframe, trade Camarilla R1/S1 breakouts from prior 12h bar with 1d EMA34 trend filter and choppiness regime (CHOP > 61.8 = range -> mean reversion at S1/R1; CHOP < 38.2 = trend -> breakout). Uses volume confirmation (>1.5x median) for signal conviction. Designed for low trade frequency (12-37/year) and robustness in bull/bear markets via regime adaptation.
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
    
    # Get 1d data for HTF trend (EMA34) and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d ATR(14) for choppiness calculation
    tr_1d = np.maximum(df_1d['high'] - df_1d['low'], 
                       np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)), 
                                  np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d TRUE RANGE and ATR for choppiness
    true_range = atr_1d
    # Sum of TRUE RANGE over 14 periods (choppiness numerator)
    sum_tr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    # Max(high) - Min(low) over 14 periods (choppiness denominator)
    max_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    # Choppiness Index: CHOP = 100 * log10(sum_TR_14 / range_14) / log10(14)
    chop_raw = np.where((range_14 > 0) & (sum_tr_14 > 0), 
                        100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 
                        50.0)  # default to neutral when invalid
    chop = np.where(np.isnan(chop_raw), 50.0, chop_raw)
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar (HLC of prior 12h)
    cam_high = pd.Series(df_12h['high'].values).shift(1).values
    cam_low = pd.Series(df_12h['low'].values).shift(1).values
    cam_close = pd.Series(df_12h['close'].values).shift(1).values
    
    # Camarilla R1, S1 levels (core breakout levels)
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    
    # Volume confirmation: volume > 1.5x median volume (30-period) for high conviction
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # ATR(12) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Align HTF indicators to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(34) 1d, chop 1d, Camarilla (need 2 bars for shift), volume median (30), ATR (12)
    start_idx = max(34, 30, 2, 30, 12) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        chop_val = chop_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        
        # Regime filters
        is_ranging = chop_val > 61.8   # choppy/range market
        is_trending = chop_val < 38.2   # trending market
        
        # Volume confirmation
        volume_spike = volume_val > 1.5 * vol_median_val
        
        if position == 0:
            # Long conditions
            if is_ranging:
                # In range: mean reversion at S1 (long near support)
                long_signal = (close_val <= s1_val * 1.005) and \
                              volume_spike and \
                              (close_val > ema_34_1d_val * 0.98)  # not too far below trend
            else:  # trending
                # In trend: breakout above R1 with momentum
                long_signal = (close_val > r1_val) and \
                              volume_spike and \
                              (close_val > ema_34_1d_val)  # uptrend confirmation
            
            # Short conditions
            if is_ranging:
                # In range: mean reversion at R1 (short near resistance)
                short_signal = (close_val >= r1_val * 0.995) and \
                               volume_spike and \
                               (close_val < ema_34_1d_val * 1.02)  # not too far above trend
            else:  # trending
                # In trend: breakdown below S1 with momentum
                short_signal = (close_val < s1_val) and \
                               volume_spike and \
                               (close_val < ema_34_1d_val)  # downtrend confirmation
            
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
            highest_since_entry = max(highest_since_entry, high_val)
            # Exit conditions: ATR trailing stop or regime/chop reversal
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            elif is_ranging and close_val > ema_34_1d_val * 1.02:  # take profit in range
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # Exit conditions: ATR trailing stop or regime/chop reversal
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            elif is_ranging and close_val < ema_34_1d_val * 0.98:  # take profit in range
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0