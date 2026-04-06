#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with 12-hour trend filter and 1-day volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13 to detect institutional buying/selling pressure.
# 12-hour EMA trend filter ensures trades align with intermediate trend.
# 1-day volume average confirms institutional participation.
# Designed for 6h timeframe to target 50-150 trades over 4 years with balanced frequency.

name = "6h_elder_ray12h_trend_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour EMA(13) for Elder Ray calculation
    close_s = pd.Series(close)
    ema_12h = close_s.ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components: Bull Power = High - EMA, Bear Power = Low - EMA
    bull_power = high - ema_12h
    bear_power = low - ema_12h
    
    # 1-day trend filter: EMA(21) for intermediate trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1-day volume average for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 21  # EMA needs 21 periods
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.2x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.2
        
        # Trend filter: price above/below daily EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: bear power turns negative or stoploss
            if (bear_power[i] > 0 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power turns negative or stoploss
            if (bull_power[i] < 0 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: bull power positive in uptrend
                if bull_power[i] > 0 and uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bear power negative in downtrend
                elif bear_power[i] < 0 and downtrend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals