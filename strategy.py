#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: On daily timeframe, Donchian(20) breakouts capture major trend moves.
1w EMA50 filter ensures alignment with weekly trend. Volume spike confirms conviction.
Choppiness index filter avoids whipsaws in sideways markets. Designed to work in both
bull and bear markets by only taking breakouts in direction of weekly trend.
Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years).
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
    
    # Get 1d data for Donchian calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1d (20-period high/low)
    # Using rolling window on 1d data, then align to 1d timeframe
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (no additional alignment needed as df_1d is already 1d)
    # But we need to shift by 1 to avoid look-ahead (use previous day's channel)
    donchian_high_shifted = np.roll(donchian_high, 1)
    donchian_low_shifted = np.roll(donchian_low, 1)
    donchian_high_shifted[0] = np.nan
    donchian_low_shifted[0] = np.nan
    
    # Calculate EMA50 on 1w close for trend
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Choppiness Index (14-period) to avoid sideways markets
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(high - low, np.absolute(np.roll(high, 1) - np.roll(low, 1)))
    tr[0] = np.nan
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr14.sum() / (np.log10(14) * (highest_high14 - lowest_low14))) if False else \
               100 * np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values / 
                              (np.log10(14) * (highest_high14 - lowest_low14)))
    # Avoid division by zero and handle NaN
    denominator = np.log10(14) * (highest_high14 - lowest_low14)
    chop = np.where((denominator > 0) & (~np.isnan(denominator)), 
                    100 * np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values / denominator), 
                    50)
    chop_filter = chop < 61.8  # Allow trading when not too choppy (below 61.8 = trending or neutral)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA, volume MA, and chop
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_shifted[i]) or np.isnan(donchian_low_shifted[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donch_high = donchian_high_shifted[i]
        donch_low = donchian_low_shifted[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        is_trending = chop_filter[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND volume spike AND price > EMA (uptrend) AND trending market
            long_entry = (curr_close > donch_high) and vol_spike and (curr_close > ema_trend) and is_trending
            # Short: price breaks below Donchian low AND volume spike AND price < EMA (downtrend) AND trending market
            short_entry = (curr_close < donch_low) and vol_spike and (curr_close < ema_trend) and is_trending
            
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
            # Exit: price crosses below Donchian low OR price crosses below EMA
            if (curr_close < donch_low) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Donchian high OR price crosses above EMA
            if (curr_close > donch_high) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "1d"
leverage = 1.0