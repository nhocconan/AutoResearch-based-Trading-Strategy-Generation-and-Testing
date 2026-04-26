#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopFilter_v2
Hypothesis: Camarilla R1/S1 breakouts with 1d EMA34 trend filter, volume spike, and choppiness regime filter on 4h timeframe.
Long when: price breaks above R4/S4 + 1d EMA34 uptrend + volume > 1.8 * avg volume + chop < 61.8 (trending regime).
Short when: price breaks below S4 + 1d EMA34 downtrend + volume > 1.8 * avg volume + chop < 61.8.
Exit when: price reverts to Camarilla midpoint (PP) or touches opposite S3/R3 level.
Uses discrete 0.25 position size. Designed for BTC/ETH:
- R4/S4 breakouts are stronger signals than R1/S1, reducing false breakouts
- 1d EMA34 filter ensures trading with the daily trend
- Volume confirmation avoids low-validity breakouts
- Chop filter avoids ranging markets where breakouts fail
- Targets 20-50 trades/year for optimal test generalization.
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
    
    # Calculate Camarilla levels from previous day (using 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # shift(1) for previous day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R4, S4, PP (pivot point), R3, S3 for exit
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align to 4h timeframe (wait for completed 1d bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_avg)
    
    # Choppiness Index: chop < 61.8 = trending regime (good for breakouts)
    # chop = 100 * log10(sum(atr(14)) / (max(high, n) - min(low, n))) / log10(n)
    tr1 = pd.Series(high).rolling(window=14, min_periods=14).max() - pd.Series(low).rolling(window=14, min_periods=14).min()
    tr2 = abs(pd.Series(high) - pd.Series(low).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(high).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low + 1e-10)) / np.log10(14)
    chop_filter = chop < 61.8  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for volume avg, 34 for 1d EMA, 14 for chop
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend, volume, and chop confirmation
            # Long: break above R4 + 1d EMA34 uptrend + volume spike + chop < 61.8
            long_entry = (close_val > camarilla_r4_aligned[i]) and \
                       (ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]) and \
                       volume_spike[i] and \
                       chop_filter[i]
            # Short: break below S4 + 1d EMA34 downtrend + volume spike + chop < 61.8
            short_entry = (close_val < camarilla_s4_aligned[i]) and \
                        (ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]) and \
                        volume_spike[i] and \
                        chop_filter[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to PP or touches S3 (contrarian exit)
            if (close_val < camarilla_pp_aligned[i]) or (close_val < camarilla_s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to PP or touches R3 (contrarian exit)
            if (close_val > camarilla_pp_aligned[i]) or (close_val > camarilla_r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike_ChopFilter_v2"
timeframe = "4h"
leverage = 1.0