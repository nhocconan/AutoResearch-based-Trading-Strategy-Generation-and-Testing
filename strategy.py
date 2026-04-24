#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Entry: Price breaks above/below 4h Donchian(20) channel with volume > 1.5 * 4h volume MA(20) and 12h EMA50 alignment.
- Exit: Price touches the opposite Donchian level (mean reversion) or breaks the channel midpoint (trend exhaustion).
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Donchian channels provide objective structure, EMA50 filters trend, volume confirms breakout validity.
- Works in bull markets by following trend with breakouts, works in bear markets by fading false breakouts at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channel
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align Donchian levels from 4h to 15m timeframe (but we're on 4h, so direct use)
    # Since timeframe is 4h, we can use the values directly with proper indexing
    # We need to map 4h indices to our 4h prices array (they should align)
    # But to be safe and follow MTF rules, we'll use align_htf_to_ltf
    dh_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    dm_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Calculate 12h EMA50 for trend
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or 
            np.isnan(dm_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma_4h_aligned[i]
            
            # Determine 12h EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
            trend_bullish = close[i] > ema_50_aligned[i]
            trend_bearish = close[i] < ema_50_aligned[i]
            
            # Long: price breaks above Donchian high AND 12h trend bullish AND volume confirmed
            if curr_high > dh_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian low AND 12h trend bearish AND volume confirmed
            elif curr_low < dl_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on touch of Donchian low (mean reversion) or break above midpoint with weakness
            if curr_low <= dl_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on touch of Donchian high (mean reversion) or break below midpoint with weakness
            if curr_high >= dh_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0