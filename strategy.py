#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h strategy using 4h Donchian breakout + volume confirmation + chop regime filter
    # Long: price > 4h Donchian upper (20) AND volume > 1.5x avg AND chop < 61.8 (trending)
    # Short: price < 4h Donchian lower (20) AND volume > 1.5x avg AND chop < 61.8 (trending)
    # Exit: price crosses Donchian midline OR volume dry-up OR chop > 61.8 (choppy)
    # Uses 4h for signal direction, 1h only for entry timing to reduce trade frequency.
    # Session filter: 08-20 UTC to avoid low-volume periods.
    # Discrete position sizing: 0.20 to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper (20-period high)
    donch_high_4h = np.full(len(high_4h), np.nan)
    for i in range(20, len(high_4h)):
        donch_high_4h[i] = np.max(high_4h[i-20:i])
    
    # Donchian lower (20-period low)
    donch_low_4h = np.full(len(low_4h), np.nan)
    for i in range(20, len(low_4h)):
        donch_low_4h[i] = np.min(low_4h[i-20:i])
    
    # Donchian midline (average of upper and lower)
    donch_mid_4h = (donch_high_4h + donch_low_4h) / 2.0
    
    # Align 4h Donchian levels to 1h
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    donch_mid_aligned = align_htf_to_ltf(prices, df_4h, donch_mid_4h)
    
    # Calculate 4h Chop Index (14-period) for regime filter
    # Chop = 100 * log10(sum(ATR(1)) / (log10(n) * ATR(14))) 
    # Simplified: high-low range based
    true_range_4h = np.maximum(
        high_4h[1:] - low_4h[1:],
        np.maximum(
            np.abs(high_4h[1:] - high_4h[:-1]),
            np.abs(low_4h[1:] - low_4h[:-1])
        )
    )
    true_range_4h = np.concatenate([[np.nan], true_range_4h])  # align with index
    
    # ATR(1) = true range
    atr1_4h = true_range_4h
    
    # Sum of ATR(1) over 14 periods
    sum_atr1_4h = np.full(len(atr1_4h), np.nan)
    for i in range(14, len(atr1_4h)):
        sum_atr1_4h[i] = np.nansum(atr1_4h[i-14:i])
    
    # ATR(14)
    atr14_4h = np.full(len(atr1_4h), np.nan)
    for i in range(14, len(atr1_4h)):
        atr14_4h[i] = np.nanmean(atr1_4h[i-14:i])
    
    # Chop Index
    chop_4h = np.full(len(atr1_4h), np.nan)
    for i in range(14, len(atr1_4h)):
        if not np.isnan(sum_atr1_4h[i]) and not np.isnan(atr14_4h[i]) and atr14_4h[i] > 0:
            log_sum = np.log10(sum_atr1_4h[i])
            log_atr_n = np.log10(atr14_4h[i] * 14)
            chop_4h[i] = 100 * (log_sum / log_atr_n)
    
    # Align 4h Chop to 1h
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop_4h)
    
    # Calculate 1h volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if not in session or data not ready
        if not in_session[i] or \
           np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or \
           np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop < 61.8 = trending (favor breakouts)
        trending_regime = chop_aligned[i] < 61.8
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian breakout + volume + regime
        long_entry = (close[i] > donch_high_aligned[i]) and vol_confirm and trending_regime
        short_entry = (close[i] < donch_low_aligned[i]) and vol_confirm and trending_regime
        
        # Exit logic: 
        # - Price crosses Donchian midline
        # - Volume dry-up
        # - Regime shifts to choppy (chop > 61.8)
        long_exit = (close[i] < donch_mid_aligned[i]) or not vol_confirm or (chop_aligned[i] >= 61.8)
        short_exit = (close[i] > donch_mid_aligned[i]) or not vol_confirm or (chop_aligned[i] >= 61.8)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_donchian_breakout_volume_chop_v1"
timeframe = "1h"
leverage = 1.0