#!/usr/bin/env python3
"""
12h_Weekly_TLDR_Pivot_Breakout_DailyTrend
Hypothesis: Uses weekly TLDR (True Low/High of the Day Range) pivot levels derived from Monday's open to Friday's close for weekly context, combined with daily trend filter (EMA34) and volume confirmation. Enters when price breaks above/below weekly pivot level in direction of daily trend with volume spike. Exits on opposite pivot level or trend reversal. Designed for 12h timeframe to capture multi-day swings while avoiding intraday noise. Targets 15-30 trades/year via strict weekly/daily confluence.
"""

name = "12h_Weekly_TLDR_Pivot_Breakout_DailyTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_tldr_pivot(weekly_open, weekly_high, weekly_low, weekly_close):
    """Calculate weekly TLDR pivot: (weekly_open + weekly_high + weekly_low + weekly_close) / 4"""
    return (weekly_open + weekly_high + weekly_low + weekly_close) / 4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly TLDR Pivot (from Monday open to Friday close) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # We need weekly OHLC from actual weekly bars
    weekly_open = df_1w['open'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate TLDR pivot for each weekly bar
    weekly_pivot = np.zeros_like(weekly_close)
    for i in range(len(weekly_close)):
        weekly_pivot[i] = calculate_weekly_tldr_pivot(
            weekly_open[i], weekly_high[i], weekly_low[i], weekly_close[i]
        )
    
    # Align weekly pivot to 12h (wait for weekly bar to close)
    weekly_pivot_12h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # --- Daily EMA34 Trend Filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume Spike Detection (24-period average for 12h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_12h[i]) or np.isnan(ema_34_12h[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: price breaks above weekly pivot + above daily EMA34 + volume
            if (close[i] > weekly_pivot_12h[i] and 
                close[i] > ema_34_12h[i] and 
                close[i-1] <= weekly_pivot_12h[i-1] and  # crossed above pivot this bar
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly pivot + below daily EMA34 + volume
            elif (close[i] < weekly_pivot_12h[i] and 
                  close[i] < ema_34_12h[i] and 
                  close[i-1] >= weekly_pivot_12h[i-1] and  # crossed below pivot this bar
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below weekly pivot OR trend turns down
                if (close[i] < weekly_pivot_12h[i] and close[i-1] >= weekly_pivot_12h[i-1]) or \
                   (close[i] < ema_34_12h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above weekly pivot OR trend turns up
                if (close[i] > weekly_pivot_12h[i] and close[i-1] <= weekly_pivot_12h[i-1]) or \
                   (close[i] > ema_34_12h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals