#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly pivot breakout with volume confirmation and 1w trend filter.
# Uses weekly pivot R4/S4 levels (extreme levels) from weekly high/low/close.
# Long when price breaks above R4 with volume > 1.5x average and 1w close > EMA34.
# Short when price breaks below S4 with volume > 1.5x average and 1w close < EMA34.
# Exit when price returns to weekly pivot (PP) or trend reverses.
# Designed for ~10-20 trades/year with strict entry conditions to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1
    # S4 = PP - (H - L) * 1.1
    pp = (high_1w + low_1w + close_1w) / 3.0
    r4 = pp + (high_1w - low_1w) * 1.1
    s4 = pp - (high_1w - low_1w) * 1.1
    
    # Get weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly pivot levels and EMA to 1d timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-period average (daily)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filters from weekly EMA34
        bullish_trend = price > ema34_aligned[i]
        bearish_trend = price < ema34_aligned[i]
        
        if position == 0:
            # Long: price breaks above R4 with volume and bullish trend
            if price > r4_aligned[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below S4 with volume and bearish trend
            elif price < s4_aligned[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below weekly pivot or trend turns bearish
            if price < pp_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above weekly pivot or trend turns bullish
            if price > pp_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyPivot_R4S4_Breakout_Volume_1wTrend"
timeframe = "1d"
leverage = 1.0