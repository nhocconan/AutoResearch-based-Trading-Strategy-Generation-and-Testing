# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation. 
# Weekly pivot levels from prior week act as support/resistance. Breakouts in the direction of 
# weekly trend (price vs weekly EMA34) with volume surge have higher follow-through. 
# Works in bull/bear: breakouts capture momentum; weekly filter avoids counter-trend whipsaws.
# Target: 15-30 trades/year (60-120 over 4 years). Size: 0.25.

#!/usr/bin/env python3
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
    
    # Get weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    r2_1w = pp_1w + (high_1w - low_1w)
    s2_1w = pp_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pp_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pp_1w)
    
    # Align weekly indicators to 6h
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # 6h Donchian channels (20-period)
    lookback = 20
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest[i] = np.max(high[i - lookback + 1:i + 1])
        lowest[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20 - 1, n):
        vol_ma[i] = np.mean(volume[i - 20 + 1:i + 1])
    vol_ratio = volume / vol_ma
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20)  # weekly EMA34 + Donchian20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(highest[i]) or 
            np.isnan(lowest[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price vs weekly EMA34
        uptrend = close[i] > ema34_1w_aligned[i]
        downtrend = close[i] < ema34_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest[i - 1]  # break above prior period's high
        breakout_down = close[i] < lowest[i - 1]  # break below prior period's low
        
        # Volume confirmation: vol > 1.5x average
        vol_confirmed = vol_ratio[i] > 1.5
        
        # Long: uptrend + upside breakout + volume
        long_condition = uptrend and breakout_up and vol_confirmed
        # Short: downtrend + downside breakout + volume
        short_condition = downtrend and breakout_down and vol_confirmed
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit: opposite breakout or loss of trend
        elif position == 1 and (breakout_down or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (breakout_up or not downtrend):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0