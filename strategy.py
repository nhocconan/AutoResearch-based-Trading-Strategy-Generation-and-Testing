#!/usr/bin/env python3
"""
1d_RegimeAdaptive_CamarillaPivot_VolumeConfirm_v1
Hypothesis: On daily timeframe, use choppiness index regime filter to switch between mean-reversion (buy Camarilla S3/S4, sell R3/R4) in ranging markets and trend-following (buy R3/S3 breakouts) in trending markets. Volume spike confirms breakout validity. This adaptive approach should work in both bull (trending) and bear (ranging/volatile) markets by aligning strategy to prevailing market regime.
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
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d choppiness index (14-period) for regime detection
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = (pd.Series(df_1d['close']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    chop = 100 * np.log10(atr_14 / (hh_14 - ll_14 + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1d EMA34 for trend filter (used in both regimes)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla pivot levels on 1d data (using previous day's OHLC)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + (camarilla_range * 1.1 / 4)
    r4 = prev_close + (camarilla_range * 1.1 / 2)
    s3 = prev_close - (camarilla_range * 1.1 / 4)
    s4 = prev_close - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla levels to 1d timeframe (no alignment needed for same TF, but use helper for consistency)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike detection on 1d (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime detection: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending (trend follow)
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        # Trend filter (EMA34)
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if is_ranging:
            # Mean-reversion regime: buy near S3/S4, sell near R3/R4
            # Long: price touches S3/S4 with volume spike
            if (close[i] <= s3_aligned[i] or close[i] <= s4_aligned[i]) and volume_spike[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Short: price touches R3/R4 with volume spike
            elif (close[i] >= r3_aligned[i] or close[i] >= r4_aligned[i]) and volume_spike[i]:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit mean-reversion position when price moves toward middle
            elif position == 1 and close[i] >= (s3_aligned[i] + r3_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] <= (r3_aligned[i] + s3_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
                    
        elif is_trending:
            # Trend-following regime: buy breakouts above R3, sell breakdowns below S3
            # Long: price breaks above R3 with volume spike + in uptrend
            if close[i] > r3_aligned[i] and volume_spike[i] and uptrend:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Short: price breaks below S3 with volume spike + in downtrend
            elif close[i] < s3_aligned[i] and volume_spike[i] and downtrend:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit trend position when price reverses to opposite level or trend weakens
            elif position == 1 and (close[i] < s3_aligned[i] or not uptrend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (close[i] > r3_aligned[i] or not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Choppy transition regime (38.2 <= chop <= 61.8): reduce activity, only strong signals
            # Only trade clear breakouts with volume spike
            if close[i] > r3_aligned[i] and volume_spike[i] and uptrend:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            elif close[i] < s3_aligned[i] and volume_spike[i] and downtrend:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            elif position == 1 and close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_RegimeAdaptive_CamarillaPivot_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0