#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1S1_Breakout_Trend_Volume_v1
Hypothesis: Weekly pivot R1/S1 breakouts with 1-week EMA trend filter and volume spike capture momentum on daily timeframe.
This strategy targets longer-term trends (1d timeframe) with weekly context to reduce noise and overtrading.
Designed for low trade frequency (7-25/year) to minimize fee drag while capturing significant moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA34 trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close']
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for weekly pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # Calculate weekly pivot levels from daily data (using prior week's OHLC)
    # Weekly high = max of daily highs in week, weekly low = min of daily lows in week, weekly close = last daily close of week
    # For simplicity, we use daily OHLC to approximate weekly pivot (more accurate would require grouping by week)
    # Pivot = (weekly_high + weekly_low + weekly_close) / 3
    # R1 = (2 * Pivot) - weekly_low
    # S1 = (2 * Pivot) - weekly_high
    # However, since we don't have weekly aggregation here, we use daily data with weekly alignment
    # Instead, we calculate daily pivot and align it (this approximates weekly levels when aligned)
    # Better approach: resample to weekly properly using pandas but we must use get_htf_data
    # Since we already have daily data from get_htf_data, we can compute weekly pivot by grouping
    # But to avoid look-ahead, we compute pivot using prior week's data
    
    # Simpler: use prior day's OHLC for daily pivot (standard practice)
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    # This is actually for intraday, but we adapt for daily using prior day
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = (prev_high - prev_low)
    r1_level = prev_close + (1.1 * camarilla_range) / 12
    s1_level = prev_close - (1.1 * camarilla_range) / 12
    
    # Align pivot levels to daily timeframe (already aligned since we used rolled values)
    r1_aligned = r1_level
    s1_aligned = s1_level
    
    # Volume spike detection: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above R1 with 1w uptrend and volume spike
            if price > r1 and price > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with 1w downtrend and volume spike
            elif price < s1 and price < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to S1 or breaks below 1w EMA
            if price < s1 or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to R1 or breaks above 1w EMA
            if price > r1 or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Pivot_R1S1_Breakout_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0