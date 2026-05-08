#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR filter and volume confirmation
# Uses daily ATR to filter volatility regimes and Donchian channel breakouts for entries.
# Works in both bull and bear markets by only taking breakouts in the direction of
# the daily trend, with volume confirmation to avoid false breakouts. Target: 20-50 trades/year.

name = "4h_Donchian_1dATR_VolumeTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period) for volatility filter
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr = np.maximum(high_daily[1:] - low_daily[1:], 
                    np.maximum(np.abs(high_daily[1:] - close_daily[:-1]),
                               np.abs(low_daily[1:] - close_daily[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR with Wilder's smoothing
    atr_14_daily = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14_daily[13] = np.nanmean(tr[1:15])  # First ATR value
        for i in range(14, len(tr)):
            atr_14_daily[i] = (atr_14_daily[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily EMA50 for trend filter
    ema50_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 50:
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = (close_daily[i] * 2 + ema50_daily[i-1] * 48) / 50
    
    # Calculate 20-period Donchian channels on 4h data
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high_20[i] = np.max(high[i-20:i])
        lowest_low_20[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_vol_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            avg_vol_20[i] = np.mean(volume[i-20:i])
    
    # Align daily indicators to 4h timeframe
    atr_14_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_14_daily)
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(atr_14_daily_aligned[i]) or np.isnan(ema50_daily_aligned[i]) or
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(avg_vol_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * avg_vol_20[i]
        
        # Volatility filter: only trade when ATR is above its 50-period average (avoid low volatility)
        # Calculate ATR 50-period average on daily timeframe
        if i >= 20:  # Ensure we have enough data for ATR calculation
            # Get current daily ATR value
            current_atr = atr_14_daily_aligned[i]
            # Calculate 50-period average of ATR (using daily values)
            # We'll approximate by checking if current ATR is above recent values
            vol_filter = True  # Simplified: always allow trading for now
            if i >= 50:  # Only apply filter after we have enough history
                # Get ATR values from the last 50 days (approximated)
                atr_values = []
                j = i
                count = 0
                while j >= 0 and count < 50:
                    if not np.isnan(atr_14_daily_aligned[j]):
                        atr_values.append(atr_14_daily_aligned[j])
                        count += 1
                    j -= 1
                if len(atr_values) >= 10:  # Minimum samples
                    atr_avg = np.nanmean(atr_values[-20:]) if len(atr_values) >= 20 else np.nanmean(atr_values)
                    vol_filter = current_atr > 0.8 * atr_avg  # Trade when ATR is not too low
        
        if position == 0:
            # Look for entry: Donchian breakout in direction of daily trend with volume confirmation
            # Long when price breaks above upper Donchian in uptrend
            long_condition = (
                close[i] > highest_high_20[i] and   # Break above Donchian high
                close[i] > ema50_daily_aligned[i] and   # Price above daily EMA50 (uptrend)
                vol_confirm and                     # Volume confirmation
                vol_filter                          # Volatility filter
            )
            
            # Short when price breaks below lower Donchian in downtrend
            short_condition = (
                close[i] < lowest_low_20[i] and    # Break below Donchian low
                close[i] < ema50_daily_aligned[i] and   # Price below daily EMA50 (downtrend)
                vol_confirm and                     # Volume confirmation
                vol_filter                          # Volatility filter
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian middle or trend changes
            donchian_middle = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] < donchian_middle or close[i] < ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian middle or trend changes
            donchian_middle = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] > donchian_middle or close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals