#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout + Volume Spike + 1d EMA34 Trend + Chop Filter
Hypothesis: Camarilla H3/L3 breakouts with volume confirmation in the direction of daily EMA34 trend, filtered by choppiness index (CHOP > 61.8 = range, CHOP < 38.2 = trend). Only trade when CHOP < 38.2 (trending regime) to avoid false breakouts in chop. Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 20-40 trades/year on 4h.
Works in bull markets via breakouts with trend and in bear markets via trend filter (avoids counter-trend entries) and chop filter (avoids whipsaws).
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
    
    # Get 1d data for pivot calculation, EMA trend, and chop filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14))
    # Simplified: CHOP = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    # We'll use ATR(14) and range over 14 periods
    tr1 = pd.Series(df_1d['high']).rolling(14, min_periods=14).max() - pd.Series(df_1d['low']).rolling(14, min_periods=14).min()
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(14, min_periods=14).sum().values
    highest_high = pd.Series(df_1d['high']).rolling(14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(14, min_periods=14).min().values
    range14 = highest_high - lowest_low
    chop = 100 * np.log10(sum_atr14 / range14) / np.log10(14)
    chop = np.where(range14 == 0, 50, chop)  # avoid div by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla pivots from previous 1d OHLC
    # H3 = close + 1.0*(high-low), L3 = close - 1.0*(high-low)
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    rang = prev_high - prev_low
    H3 = prev_close + 1.0 * rang
    L3 = prev_close - 1.0 * rang
    
    # Align Camarilla levels to 4h (use previous day's levels for current day's trading)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA, EMA, and chop
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        H3_level = H3_aligned[i]
        L3_level = L3_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        # Only trade in trending regime: CHOP < 38.2
        trending_regime = chop_val < 38.2
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND price > 1d EMA34 (uptrend) AND trending regime
            long_entry = (curr_close > H3_level) and vol_spike and (curr_close > ema_trend) and trending_regime
            # Short: price breaks below L3 AND volume spike AND price < 1d EMA34 (downtrend) AND trending regime
            short_entry = (curr_close < L3_level) and vol_spike and (curr_close < ema_trend) and trending_regime
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below L3 (reversal) OR price < 1d EMA34 (trend change) OR chop > 61.8 (chop regime)
            if (curr_close < L3_level) or (curr_close < ema_trend) or (chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 (reversal) OR price > 1d EMA34 (trend change) OR chop > 61.8 (chop regime)
            if (curr_close > H3_level) or (curr_close > ema_trend) or (chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_VolumeSpike_1dEMA34_Trend_ChopFilter"
timeframe = "4h"
leverage = 1.0