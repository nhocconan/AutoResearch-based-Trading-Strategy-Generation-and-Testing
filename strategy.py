#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA filter and volume confirmation
# Uses 4h Donchian channel breakout (20-period) for trend direction, filtered by daily EMA50 trend and volume > 1.5x 20-period average.
# Designed to capture breakouts in trending markets while avoiding false breakouts in ranging markets.
# Target: 20-50 trades/year (80-200 total over 4 years). Works in bull/bear via EMA trend filter.

name = "4h_Donchian20_1dEMA50_VolumeConfirm"
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
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channel on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 20-period high and low for Donchian channels
    donchian_high = np.full(len(high_4h), np.nan)
    donchian_low = np.full(len(low_4h), np.nan)
    
    for i in range(20, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
    
    # Get daily data for EMA50 trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend
    close_daily = df_daily['close'].values
    ema50_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 50:
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = (close_daily[i] * 2 + ema50_daily[i-1] * 48) / 50
    
    # Calculate 4h volume average for volume filter
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align Donchian high/low to 4h timeframe (already aligned as we're using 4h data)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Align daily EMA50 to 4h timeframe
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
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(ema50_daily_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current 4h bar's Donchian levels (last completed 4h bar)
        donchian_high_current = np.nan
        donchian_low_current = np.nan
        if not np.isnan(donchian_high_aligned[i]) and not np.isnan(donchian_low_aligned[i]):
            idx_4h = 0
            while idx_4h < len(df_4h) and df_4h.iloc[idx_4h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_4h += 1
            idx_4h -= 1  # last completed 4h bar
            
            if idx_4h >= 0:
                donchian_high_current = donchian_high_aligned[i]
                donchian_low_current = donchian_low_aligned[i]
        
        # Get current daily bar's EMA50 (last completed daily bar)
        ema50_daily_current = np.nan
        if not np.isnan(ema50_daily_aligned[i]):
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                ema50_daily_current = ema50_daily_aligned[i]
        
        if np.isnan(donchian_high_current) or np.isnan(donchian_low_current) or np.isnan(ema50_daily_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        price_above_donchian_high = close[i] > donchian_high_current
        price_below_donchian_low = close[i] < donchian_low_current
        price_above_ema = close[i] > ema50_daily_current
        price_below_ema = close[i] < ema50_daily_current
        vol_filter = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with EMA trend filter and volume confirmation
            # Long when price breaks above Donchian high in uptrend, short when breaks below Donchian low in downtrend
            if price_above_donchian_high and price_above_ema and vol_filter:
                signals[i] = 0.25
                position = 1
            elif price_below_donchian_low and price_below_ema and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian low or trend fails or volume drops
            if close[i] < donchian_low_current or not price_above_ema or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian high or trend fails or volume drops
            if close[i] > donchian_high_current or not price_below_ema or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals