# 1d_Weekly_Pullback_Trend_Follow
# Hypothesis: On daily timeframe, buy pullbacks to weekly EMA21 in uptrend (price > weekly EMA50),
# sell/short pullbacks to weekly EMA21 in downtrend (price < weekly EMA50).
# Uses weekly trend filter with daily entry on pullbacks to reduce whipsaw.
# Weekly EMA21 acts as dynamic support/resistance. Designed for fewer trades (~10-20/year)
# and works in both bull (buy pullbacks in uptrend) and bear (sell pullbacks in downtrend).
# Volume confirmation filters low-momentum breakouts.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 21:
        return np.zeros(n)
    
    # Weekly EMA21 and EMA50 for trend and dynamic support/resistance
    weekly_close = df_weekly['close'].values
    ema21_weekly = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to daily timeframe (already waits for weekly bar close)
    ema21_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema21_weekly)
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Daily average volume (20-period) for confirmation
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    # Start after weekly EMA50 warmup
    start = 50
    for i in range(start, n):
        if (np.isnan(ema21_weekly_aligned[i]) or np.isnan(ema50_weekly_aligned[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        ema21 = ema21_weekly_aligned[i]
        ema50 = ema50_weekly_aligned[i]
        
        if position == 0:
            # Long setup: price > weekly EMA50 (uptrend) and pulling back to weekly EMA21
            if price > ema50 and price <= ema21 * 1.01 and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short setup: price < weekly EMA50 (downtrend) and pulling back to weekly EMA21
            elif price < ema50 and price >= ema21 * 0.99 and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below weekly EMA21 or above weekly EMA50 (trend change)
            if price < ema21 * 0.99 or price > ema50 * 1.01:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above weekly EMA21 or below weekly EMA50 (trend change)
            if price > ema21 * 1.01 or price < ema50 * 0.99:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Weekly_Pullback_Trend_Follow"
timeframe = "1d"
leverage = 1.0