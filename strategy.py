#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Strategy: Daily Camarilla pivot levels with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla levels act as strong support/resistance. Weekly trend filter ensures trades align with higher timeframe direction. Volume > 1.5x 20-day average confirms institutional participation. Designed for low trade frequency (~10-25/year) to minimize fee drift. Works in bull markets via long entries at support and bear markets via short entries at resistance.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily Camarilla pivot levels (based on previous day's range)
    # Calculate pivot and levels for each day using previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First day has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Pivot point = (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Range = H - L
    rng = prev_high - prev_low
    
    # Camarilla levels
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    r1 = close + rng * 1.1 / 12.0
    r2 = close + rng * 1.1 / 6.0
    r3 = close + rng * 1.1 / 4.0
    r4 = close + rng * 1.1 / 2.0
    s1 = close - rng * 1.1 / 12.0
    s2 = close - rng * 1.1 / 6.0
    s3 = close - rng * 1.1 / 4.0
    s4 = close - rng * 1.1 / 2.0
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Weekly trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_50_1w_aligned[i]
        trend_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        # Long: Price touches S1 or S2 (support) AND bullish trend AND volume confirmation
        if ((low[i] <= s1[i] and close[i] > s1[i]) or (low[i] <= s2[i] and close[i] > s2[i])) and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price touches R1 or R2 (resistance) AND bearish trend AND volume confirmation
        elif ((high[i] >= r1[i] and close[i] < r1[i]) or (high[i] >= r2[i] and close[i] < r2[i])) and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price touches opposite level (R3/S3 for profit taking) or reverse signal
        elif position == 1 and ((high[i] >= r3[i] and close[i] < r3[i]) or (high[i] >= r4[i] and close[i] < r4[i])):
            position = 0
            signals[i] = 0.0
        elif position == -1 and ((low[i] <= s3[i] and close[i] > s3[i]) or (low[i] <= s4[i] and close[i] > s4[i])):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals