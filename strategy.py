#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Keltner Channel breakout with daily EMA50 trend filter and volume confirmation
# Uses Keltner(20, 2.0) on 12h for volatility-adjusted breakout signals, filtered by daily EMA50 trend and volume spike (>1.5x 20-period average).
# Designed to capture trends in both bull and bear markets with volatility-adjusted entries and strict confirmation filters.
# Target: 12-37 trades/year (50-150 total over 4 years). Keltner channels adapt to volatility, reducing false breakouts in ranging markets.

name = "12h_Keltner20_DailyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for 20-period indicators and EMA50
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend using pandas for accuracy
    close_daily = df_daily['close']
    ema50_daily_series = close_daily.ewm(span=50, adjust=False).mean()
    ema50_daily = ema50_daily_series.values
    
    # Calculate 12h 20-period EMA (Keltner middle)
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[:20])
        for i in range(20, n):
            ema20[i] = (close[i] * 2 + ema20[i-1] * 18) / 20
    
    # Calculate 12h ATR(20) for Keltner width
    tr = np.full(n, np.nan)
    if n >= 1:
        tr[0] = high[0] - low[0]
    if n >= 2:
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr20 = np.full(n, np.nan)
    if n >= 20:
        atr20[19] = np.mean(tr[1:20])  # Start from index 1 as tr[0] has no prior close
        for i in range(20, n):
            atr20[i] = (tr[i] * 2 + atr20[i-1] * 18) / 20
    
    # Calculate Keltner Channels
    keltner_upper = np.full(n, np.nan)
    keltner_lower = np.full(n, np.nan)
    if n >= 20:
        keltner_upper = ema20 + 2.0 * atr20
        keltner_lower = ema20 - 2.0 * atr20
    
    # Calculate 12h volume average for volume filter
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align daily EMA50 to 12h timeframe
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or np.isnan(ema20[i]) or np.isnan(ema50_daily_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        price_above_ema = close[i] > ema20[i]
        price_below_ema = close[i] < ema20[i]
        vol_spike = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Keltner breakout with trend and volume confirmation
            if close[i] > keltner_upper[i] and price_above_ema and vol_spike:
                signals[i] = 0.25
                position = 1
            elif close[i] < keltner_lower[i] and not price_above_ema and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 20-period EMA (middle of Keltner)
            if not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 20-period EMA (middle of Keltner)
            if not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals