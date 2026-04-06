#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index combined with 12-hour trend filter and 1-week volume confirmation.
# Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures bull/bear strength.
# 12-hour EMA50 filters for trend direction to avoid counter-trend whipsaws.
# Weekly volume surge (>2x average) confirms institutional participation.
# Designed for 6h timeframe to target 100-200 trades over 4 years with controlled frequency.

name = "6h_elder_ray12h_trend_vol_v1"
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
    
    # 12-hour EMA50 for trend direction
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 13-period EMA for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=5, min_periods=1).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # EMA50 needs 50 periods
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 2x weekly average (institutional participation)
        volume_filter = volume[i] > vol_ma_1w_aligned[i] * 2.0
        
        # Trend filter: EMA50 direction
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: bear power turning negative or stoploss
            if (bear_power[i] > -0.1 * close[i] or  # Bear power weakening
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power turning positive or stoploss
            if (bull_power[i] < 0.1 * close[i] or  # Bull power weakening
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: strong bull power in uptrend
                if bull_power[i] > 0.3 * close[i] and uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: strong bear power in downtrend
                elif bear_power[i] < -0.3 * close[i] and downtrend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals