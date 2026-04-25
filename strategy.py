#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout + Volume Spike + 1d EMA34 Trend Filter + Chop Filter
Hypothesis: Camarilla pivot levels act as support/resistance. Breakouts above H3 or below L3 with volume confirmation indicate institutional participation. 1d EMA34 filter ensures trades align with daily trend. Chop filter avoids whipsaws in ranging markets. Works in bull via trend-aligned breakouts and in bear via trend filter (avoids counter-trend entries). Discrete sizing minimizes fee churn.
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
    
    # Get 1d data for pivot calculation and EMA trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivots from previous 1d OHLC
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
    
    # Calculate Choppiness Index (14-period) on 4h to filter ranging markets
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar TR = high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14.sum() / np.log10(hh14 - ll14) / np.log10(14)) if (hh14 - ll14) > 0 else 50
    # Vectorized chop calculation
    atr14_series = pd.Series(tr)
    atr14_rolled = atr14_series.rolling(window=14, min_periods=14).mean().values
    hh14_series = pd.Series(high)
    ll14_series = pd.Series(low)
    hh14_rolled = hh14_series.rolling(window=14, min_periods=14).max().values
    ll14_rolled = ll14_series.rolling(window=14, min_periods=14).min().values
    sum_atr14 = pd.Series(atr14_rolled).rolling(window=14, min_periods=1).sum().values
    range14 = hh14_rolled - ll14_rolled
    chop = np.where(range14 > 0, 100 * np.log10(sum_atr14 / np.log10(range14) / np.log10(14)), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA, EMA, and chop
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        H3_level = H3_aligned[i]
        L3_level = L3_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        
        # Only trade in trending markets (CHOP < 38.2) or strong breakouts
        trending_market = chop_val < 38.2
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND price > 1d EMA34 (uptrend) AND trending market
            long_entry = (curr_close > H3_level) and vol_spike and (curr_close > ema_trend) and trending_market
            # Short: price breaks below L3 AND volume spike AND price < 1d EMA34 (downtrend) AND trending market
            short_entry = (curr_close < L3_level) and vol_spike and (curr_close < ema_trend) and trending_market
            
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
            # Exit: price crosses below L3 (reversal) OR price < 1d EMA34 (trend change) OR chop > 61.8 (strong ranging)
            if (curr_close < L3_level) or (curr_close < ema_trend) or (chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 (reversal) OR price > 1d EMA34 (trend change) OR chop > 61.8 (strong ranging)
            if (curr_close > H3_level) or (curr_close > ema_trend) or (chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_VolumeSpike_1dEMA34_Trend_ChopFilter"
timeframe = "4h"
leverage = 1.0