#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout + 1d EMA34 trend filter + volume spike + choppiness regime filter.
# Long when price breaks above R3 with 1d uptrend, volume spike, and choppy market (CHOP > 61.8).
# Short when price breaks below S3 with 1d downtrend, volume spike, and choppy market.
# Exit when price reverts to R2/S2 or trend changes.
# Uses 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years).
# Camarilla pivot levels provide institutional support/resistance, EMA34 filters higher-timeframe trend,
# volume spike confirms institutional participation, chop filter avoids whipsaws in strong trends.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, R2, S2, S3)
    # Based on previous day's OHLC
    df_1d_prev = df_1d.shift(1)
    high_prev = df_1d_prev['high'].values
    low_prev = df_1d_prev['low'].values
    close_prev = df_1d_prev['close'].values
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    R3 = pivot + (range_prev * 1.1 / 4)
    R2 = pivot + (range_prev * 1.1 / 2)
    S2 = pivot - (range_prev * 1.1 / 2)
    S3 = pivot - (range_prev * 1.1 / 4)
    
    # Align 1d Camarilla levels to 4h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Calculate 4h choppiness index (CHOP) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = true_range(high, low, prev_close)
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10(atr_14 * np.sqrt(14) / chop_denom) / np.log10(14)
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(R2_aligned[i]) or 
            np.isnan(S2_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        
        # Volume spike condition: current 4h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 2.0)
        
        # Choppiness regime filter: CHOP > 61.8 indicates ranging/choppy market (good for mean reversion)
        choppy_market = chop[i] > 61.8
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Price breaks above R3 AND 1d uptrend AND volume spike AND choppy market AND session
            if close_val > R3_aligned[i] and trend_up and volume_spike and choppy_market:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 AND 1d downtrend AND volume spike AND choppy market AND session
            elif close_val < S3_aligned[i] and trend_down and volume_spike and choppy_market:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price reverts to R2 OR trend changes to downtrend
            if close_val < R2_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price reverts to S2 OR trend changes to uptrend
            if close_val > S2_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals