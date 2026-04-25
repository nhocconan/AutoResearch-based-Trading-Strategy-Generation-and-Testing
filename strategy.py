#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: On daily timeframe, Donchian breakouts capture strong trends. 
1w EMA50 filters for higher timeframe trend alignment. 
Volume spike confirms breakout strength. 
Choppiness index filter avoids whipsaw in ranging markets.
Designed for BTC/ETH with 30-100 total trades over 4 years to minimize fee drag.
Works in both bull (breakouts with trend) and bear (filters prevent counter-trend entries).
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
    
    # Get 1d data for Donchian, EMA50, and chop calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need enough for 20-period Donchian + 50 EMA + chop
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    donchian_high = high_1d.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1d.rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 1w data for higher timeframe trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        close_1w = pd.Series(df_1w['close'])
        ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # Calculate 1d choppiness index (14-period) for regime filter
    # Chop = 100 * log10(sum(ATR14) / (log10(highest-highest-lowest-lowest) / log10(14)))
    # Simplified: Chop = 100 * log10( sum(ATR14) / (HHV - LLV) ) / log10(14)
    # We'll use: Chop > 61.8 = ranging, Chop < 38.2 = trending
    tr_1d = np.maximum(
        high_1d.values[1:] - low_1d.values[:-1],
        np.maximum(
            np.abs(high_1d.values[1:] - close_1d.values[:-1]),
            np.abs(low_1d.values[1:] - close_1d.values[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align length
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    hh_14 = high_1d.rolling(window=14, min_periods=14).max().values
    ll_14 = low_1d.rolling(window=14, min_periods=14).min().values
    chop_denom = hh_14 - ll_14
    chop_denom_safe = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_ratio = np.sum(pd.DataFrame(atr_14).rolling(14, min_periods=14).sum().values.reshape(-1, 14), axis=1) / chop_denom_safe
    chop_1d = 100 * np.log10(chop_ratio) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (1d)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(50, 20)  # 50 for EMA50, 20 for Donchian/chop/vol MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1d_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        chop_val = chop_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price above/both EMAs for uptrend, below/both for downtrend
        uptrend = curr_close > ema_50_val and curr_close > ema_50_1w_val
        downtrend = curr_close < ema_50_val and curr_close < ema_50_1w_val
        
        # Chop filter: only trade when market is trending (Chop < 38.2)
        chop_filter = chop_val < 38.2
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            if uptrend and chop_filter:
                # Uptrend: look for long breakout above Donchian high with volume
                long_signal = (curr_close > donchian_high_val) and volume_confirm
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
                    position = 0
            elif downtrend and chop_filter:
                # Downtrend: look for short breakdown below Donchian low with volume
                short_signal = (curr_close < donchian_low_val) and volume_confirm
                if short_signal:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
                    position = 0
            else:
                # No clear trend or choppy market: stay flat
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below Donchian low OR trend changes
            if curr_close < donchian_low_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high OR trend changes
            if curr_close > donchian_high_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "1d"
leverage = 1.0