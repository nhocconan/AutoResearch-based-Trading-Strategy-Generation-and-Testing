#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA Crossover with 4h Trend Filter and 1d Volume Regime
# Uses fast/slow EMA crossovers on 1h for entry timing, filtered by 4h EMA trend direction
# and 1d volume regime (high volume = active market). Avoids low-volume, choppy periods.
# Works in bull/bear by only taking trades in direction of higher timeframe trend.
# Target: 60-150 total trades over 4 years (15-37/year) with session filter (08-20 UTC)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate indicators on HTF data
    # 4h EMA for trend direction (20-period)
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d average volume for regime filter (20-period)
    vol_1d = pd.Series(df_1d['volume'].values)
    avg_vol_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # 1h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h EMA for entry signals (9 and 21 period)
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 30  # for EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume regime filter: only trade when 1d volume > average volume
        if volume[i] < 0.5 * avg_vol_1d_aligned[i]:  # Avoid very low volume periods
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: fast EMA crosses above slow EMA AND price above 4h EMA (uptrend)
            if ema_fast[i] > ema_slow[i] and close[i] > ema_4h_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: fast EMA crosses below slow EMA AND price below 4h EMA (downtrend)
            elif ema_fast[i] < ema_slow[i] and close[i] < ema_4h_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: fast EMA crosses below slow EMA
            if ema_fast[i] < ema_slow[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: fast EMA crosses above slow EMA
            if ema_fast[i] > ema_slow[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_EMA_Cross_4hTrend_1dVolRegime"
timeframe = "1h"
leverage = 1.0