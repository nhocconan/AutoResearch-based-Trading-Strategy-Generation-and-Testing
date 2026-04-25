#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeRegime_ChopFilter
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter, volume confirmation, and choppiness regime filter.
Long when price breaks above R3 in 1d uptrend with volume > 1.5x 20-period average and chop < 61.8.
Short when price breaks below S3 in 1d downtrend with volume > 1.5x 20-period average and chop < 61.8.
Exit via ATR trailing stop (2.5*ATR from extreme) or re-entry into Camarilla H3/L3 range.
Designed for ~12-25 trades/year by requiring multiple confluence conditions.
Works in bull/bear markets via 1d EMA50 filter; avoids whipsaws via volume and chop filters.
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
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for Camarilla levels
    atr_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate Camarilla levels for 1d (based on previous day)
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first period
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2
    s3 = prev_close - 1.1 * camarilla_range / 2
    h3 = prev_close + 1.1 * camarilla_range / 4
    l3 = prev_close - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate 12h ATR(14) for trailing stop
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    # Choppiness Index (CHOP) on 12h - range: 0-100
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    chop_period = 14
    atr_12h_for_chop = pd.Series(tr_12h).rolling(window=chop_period, min_periods=chop_period).mean().values
    sum_atr = pd.Series(atr_12h_for_chop).rolling(window=chop_period, min_periods=chop_period).sum().values
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    range_chop = max_high - min_low
    # Avoid division by zero
    range_chop = np.where(range_chop == 0, 1e-10, range_chop)
    chop = 100 * (np.log10(sum_atr) - np.log10(range_chop)) / np.log10(chop_period)
    # CHOP < 38.2 = trending, CHOP > 61.8 = ranging
    chop_regime = chop < 61.8  # Allow trading in trending markets (CHOP < 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0   # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, 50, atr_period, chop_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_12h[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1d EMA50 filter + CHOP < 61.8)
            if close[i] > ema_trend and chop_regime[i]:  # 1d uptrend regime
                # Long: break above R3 with volume spike
                long_signal = (close[i] > r3_aligned[i]) and vol_regime[i]
            elif close[i] < ema_trend and chop_regime[i]:  # 1d downtrend regime
                # Short: break below S3 with volume spike
                short_signal = (close[i] < s3_aligned[i]) and vol_regime[i]
            else:
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: ATR trailing stop OR re-enter Camarilla H3/L3 range
            atr_stop = long_extreme - 2.5 * atr_12h[i]
            range_exit = (close[i] < h3_aligned[i] and close[i] > l3_aligned[i])
            if close[i] <= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions: ATR trailing stop OR re-enter Camarilla H3/L3 range
            atr_stop = short_extreme + 2.5 * atr_12h[i]
            range_exit = (close[i] > l3_aligned[i] and close[i] < h3_aligned[i])
            if close[i] >= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeRegime_ChopFilter"
timeframe = "12h"
leverage = 1.0