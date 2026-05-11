#!/usr/bin/env python3
# 1d_PriceChannelBreakout_1wTrend
# Hypothesis: Uses 1d Donchian breakout for entries, filtered by 1w trend direction and volume confirmation.
# Long when: price breaks above 1d Donchian upper channel (20), weekly trend is up (price > weekly EMA20), and volume > 1.5x 20-day average.
# Short when: price breaks below 1d Donchian lower channel (20), weekly trend is down (price < weekly EMA20), and volume > 1.5x 20-day average.
# Exit when price returns to the 10-day Donchian midpoint (mean reversion exit).
# Designed to capture medium-term trends with clear entry/exit rules, avoiding false breakouts in low-volume conditions.
# Works in bull markets by catching upward breakouts and in bear markets by catching downward breakouts.

name = "1d_PriceChannelBreakout_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter (weekly EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Donchian channels (20-period) ---
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    midpoint = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
        midpoint[i] = (upper[i] + lower[i]) / 2.0
    
    # --- 1w EMA20 for trend filter ---
    close_1w = df_1w['close'].values
    ema_20 = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        if i == 20:
            ema_20[i] = np.mean(close_1w[:20])
        else:
            ema_20[i] = (close_1w[i] * 2/21) + (ema_20[i-1] * (1 - 2/21))
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1w EMA20 to 1d timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Donchian(20), EMA20, and volume MA(20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(midpoint[i]) or
            np.isnan(ema_20_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from 1w: price above/below weekly EMA20
        is_uptrend = close[i] > ema_20_aligned[i]
        is_downtrend = close[i] < ema_20_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: weekly uptrend + volume spike + break above 1d Donchian upper
                if close[i] > upper[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: weekly downtrend + volume spike + break below 1d Donchian lower
                if close[i] < lower[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price returns to 10-day Donchian midpoint (mean reversion)
                if close[i] <= midpoint[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to 10-day Donchian midpoint (mean reversion)
                if close[i] >= midpoint[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals