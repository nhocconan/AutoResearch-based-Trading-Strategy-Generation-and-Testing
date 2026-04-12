# 1d_1w_Camarilla_Breakout_Trend_v1
# Hypothesis: Weekly trend direction from 1w timeframe filters Camarilla breakout signals on 1d chart.
# In bull markets, only take long signals from Camarilla support; in bear markets, only take short signals from resistance.
# Uses weekly EMA(21) for trend filter to avoid counter-trend trades.
# Target: 15-25 trades per year (60-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Breakout_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === WEEKLY TREND FILTER (EMA21) ===
    weekly_close = df_1w['close'].values
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_trend_up = weekly_ema21 > np.roll(weekly_ema21, 1)  # Rising EMA = uptrend
    weekly_trend_down = weekly_ema21 < np.roll(weekly_ema21, 1)  # Falling EMA = downtrend
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # === DAILY CAMARILLA PIVOT LEVELS ===
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate from previous daily bar's OHLC
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close = np.roll(daily_close, 1)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    l3 = pivot + (range_val * 1.1 / 4)
    l4 = pivot + (range_val * 1.1 / 2)
    h3 = pivot - (range_val * 1.1 / 4)
    h4 = pivot - (range_val * 1.1 / 2)
    
    # Align to daily timeframe
    l3_d = align_htf_to_ltf(prices, df_1d, l3)
    l4_d = align_htf_to_ltf(prices, df_1d, l4)
    h3_d = align_htf_to_ltf(prices, df_1d, h3)
    h4_d = align_htf_to_ltf(prices, df_1d, h4)
    
    # === VOLUME CONFIRMATION (1.5x 20-day average) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(l3_d[i]) or np.isnan(l4_d[i]) or 
            np.isnan(h3_d[i]) or np.isnan(h4_d[i]) or
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price breaks Camarilla levels with close
        break_l3 = close[i] < l3_d[i]  # Break below support
        break_l4 = close[i] < l4_d[i]  # Break below stronger support
        break_h3 = close[i] > h3_d[i]  # Break above resistance
        break_h4 = close[i] > h4_d[i]  # Break above stronger resistance
        
        # Entry conditions: trend-aligned breakouts with volume
        long_entry = (break_l3 or break_l4) and weekly_trend_up_aligned[i] and vol_spike[i]
        short_entry = (break_h3 or break_h4) and weekly_trend_down_aligned[i] and vol_spike[i]
        
        # Exit conditions: reverse break of opposite level
        long_exit = close[i] > h3_d[i]  # Exit long if breaks above H3
        short_exit = close[i] < l3_d[i]  # Exit short if breaks below L3
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals