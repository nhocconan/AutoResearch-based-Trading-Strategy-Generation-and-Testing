#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Keltner Channel breakout with 1-week trend filter and volume confirmation.
# Long when close breaks above upper KC(20,2) with weekly EMA50 uptrend and volume > 1.5x average.
# Short when close breaks below lower KC(20,2) with weekly EMA50 downtrend and volume > 1.5x average.
# Exit when close crosses back through the middle line (EMA20).
# Uses Keltner Channels for volatility-based breakouts, targeting 15-30 trades per year.
# Works in bull (breakouts with trend) and bear (mean reversion in ranges via exit).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_weekly_period = 50
    ema_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= ema_weekly_period:
        ema_weekly[ema_weekly_period - 1] = np.mean(close_weekly[:ema_weekly_period])
        for i in range(ema_weekly_period, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * (2 / (ema_weekly_period + 1)) + 
                             ema_weekly[i - 1] * (1 - (2 / (ema_weekly_period + 1))))
    
    # Align weekly EMA to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate daily Keltner Channels (20,2)
    kc_period = 20
    kc_multiplier = 2.0
    
    # EMA20 for middle line
    ema20 = np.full(n, np.nan)
    if n >= kc_period:
        ema20[kc_period - 1] = np.mean(close[:kc_period])
        for i in range(kc_period, n):
            ema20[i] = (close[i] * (2 / (kc_period + 1)) + 
                        ema20[i - 1] * (1 - (2 / (kc_period + 1))))
    
    # ATR for channel width
    atr_period = 20
    tr = np.full(n, np.nan)
    atr = np.full(n, np.nan)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    for i in range(atr_period, n):
        if i == atr_period:
            atr[i] = np.mean(tr[1:atr_period+1])
        else:
            atr[i] = (tr[i] * (2 / (atr_period + 1)) + 
                      atr[i-1] * (1 - (2 / (atr_period + 1))))
    
    # Upper and lower KC bands
    kc_upper = ema20 + (kc_multiplier * atr)
    kc_lower = ema20 - (kc_multiplier * atr)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA20, ATR, weekly EMA50, and volume MA20
    start_idx = max(kc_period, atr_period, ema_weekly_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema20[i]) or np.isnan(atr[i]) or 
            np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above upper KC with weekly uptrend and volume
            if (price > kc_upper[i] and 
                price > ema_weekly_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below lower KC with weekly downtrend and volume
            elif (price < kc_lower[i] and 
                  price < ema_weekly_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below middle line (EMA20)
            if price < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above middle line (EMA20)
            if price > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KeltnerBreakout_WeeklyEMA50_Volume"
timeframe = "1d"
leverage = 1.0