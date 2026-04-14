#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian breakout + volume confirmation.
# In choppy markets (Chop > 61.8), mean-reversion at Donchian bands works; in trending (Chop < 38.2), breakouts work.
# This adapts to both bull and bear markets by filtering regime. Target: 20-40 trades/year.
# Uses 4h primary timeframe, 1d for Chop and Donchian, volume confirmation on 4h.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily True Range for Chop (14-period)
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate Chop: 100 * log10(sum(TR,14) / (max(HH,14) - min(LL,14))) / log10(14)
    sum_tr_14 = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        sum_tr_14[13] = np.sum(tr[:14])
        for i in range(14, len(df_1d)):
            sum_tr_14[i] = sum_tr_14[i-1] - tr[i-14] + tr[i]
    
    max_high_14 = np.full(len(df_1d), np.nan)
    min_low_14 = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        for i in range(13, len(df_1d)):
            max_high_14[i] = np.max(high_1d[i-13:i+1])
            min_low_14[i] = np.min(low_1d[i-13:i+1])
    
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if max_high_14[i] > min_low_14[i] and sum_tr_14[i] > 0:
            chop_1d[i] = 100 * np.log10(sum_tr_14[i] / (max_high_14[i] - min_low_14[i])) / np.log10(14)
        else:
            chop_1d[i] = 50  # neutral
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate daily Donchian channels (20-period)
    donch_high_1d = np.full(len(df_1d), np.nan)
    donch_low_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        for i in range(19, len(df_1d)):
            donch_high_1d[i] = np.max(high_1d[i-19:i+1])
            donch_low_1d[i] = np.min(low_1d[i-19:i+1])
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Calculate 4h volume moving average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):  # start after warmup
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_1d[13] > 0:  # ensure ATR is valid
            atr_val = atr_1d[min(i // 16 + 1, len(atr_1d)-1)] if i // 16 + 1 < len(atr_1d) else atr_1d[-1]
            if atr_val < 0.003 * close[i]:
                signals[i] = 0.0
                continue
        
        # Volume ratio: current volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        vol_threshold = 2.0  # require volume spike
        
        if position == 0:
            # Chop > 61.8: range -> mean reversion at Donchian bands
            # Chop < 38.2: trend -> breakout
            if chop_aligned[i] > 61.8:
                # Mean reversion: sell at Donchian high, buy at Donchian low
                if close[i] < donch_low_aligned[i] and volume_ratio > vol_threshold:
                    position = 1
                    signals[i] = position_size
                elif close[i] > donch_high_aligned[i] and volume_ratio > vol_threshold:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif chop_aligned[i] < 38.2:
                # Trend following: breakout in direction of trend
                if close[i] > donch_high_aligned[i] and volume_ratio > vol_threshold:
                    position = 1
                    signals[i] = position_size
                elif close[i] < donch_low_aligned[i] and volume_ratio > vol_threshold:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                # Neutral chop: no trade
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses Donchian high (for trend) or low (for mean reversion)
            # In trend: exit on breakout failure; in mean reversion: exit at opposite band
            if chop_aligned[i] > 61.8:
                # Mean reversion mode: exit at Donchian high
                if close[i] > donch_high_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # Trend mode: exit if price fails to hold above Donchian low
                if close[i] < donch_low_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
        elif position == -1:
            # Short exit: price crosses Donchian low (for trend) or high (for mean reversion)
            if chop_aligned[i] > 61.8:
                # Mean reversion mode: exit at Donchian low
                if close[i] < donch_low_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # Trend mode: exit if price fails to hold below Donchian high
                if close[i] > donch_high_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
    
    return signals

name = "4h_1d_Chop_Donchian_MeanRev_Trend"
timeframe = "4h"
leverage = 1.0