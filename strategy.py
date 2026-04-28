#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h Supertrend filter and volume confirmation.
# Enter long when price breaks above Camarilla R3 level with 4h Supertrend bullish and volume > 1.5x 20-bar average.
# Enter short when price breaks below Camarilla S3 level with 4h Supertrend bearish and volume confirmation.
# Exit when price retraces to Camarilla H3/L3 levels.
# Uses 4h Supertrend for trend alignment (works in both bull and bear) and 1h for precise entry timing.
# Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years.

name = "1h_Camarilla_R3S3_Breakout_4hSupertrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Supertrend
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (ATR=10, multiplier=3)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR (Wilder's smoothing)
    atr_period = 10
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period + 1:
        atr[atr_period] = np.nanmean(tr[1:atr_period+1])
        for i in range(atr_period + 1, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend calculation
    multiplier = 3
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = np.full_like(close_4h, np.nan)
    uptrend = np.full_like(close_4h, True)
    
    for i in range(1, len(close_4h)):
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            supertrend[i] = np.nan
            uptrend[i] = uptrend[i-1] if i > 0 else True
            continue
            
        if close_4h[i] <= upper_band[i-1]:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
        else:
            upper_band[i] = upper_band[i]
            
        if close_4h[i] >= lower_band[i-1]:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        else:
            lower_band[i] = lower_band[i]
            
        if supertrend[i-1] == upper_band[i-1]:
            supertrend[i] = lower_band[i] if close_4h[i] <= lower_band[i] else upper_band[i]
            uptrend[i] = close_4h[i] > lower_band[i]
        else:
            supertrend[i] = upper_band[i] if close_4h[i] >= upper_band[i] else lower_band[i]
            uptrend[i] = close_4h[i] >= upper_band[i]
    
    # Align Supertrend to 1h
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    uptrend_aligned = align_htf_to_ltf(prices, df_4h, uptrend.astype(float))
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align to 1h
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla levels (R3/S3 for entry, H3/L3 for exit)
    R3 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    S3 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    H3 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 6
    L3 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 6
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Ensure sufficient history for volume MA and HTF data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (not in_session[i] or np.isnan(supertrend_aligned[i]) or np.isnan(uptrend_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(prev_close_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(H3[i]) or np.isnan(L3[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 4h Supertrend: uptrend_aligned == 1 for bullish, == 0 for bearish
        is_uptrend = uptrend_aligned[i] == 1
        is_downtrend = uptrend_aligned[i] == 0
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > R3, 4h uptrend, volume confirm
            if price > R3[i] and is_uptrend and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: price < S3, 4h downtrend, volume confirm
            elif price < S3[i] and is_downtrend and vol_confirm:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at H3
            if price <= H3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - hold or exit at L3
            if price >= L3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals