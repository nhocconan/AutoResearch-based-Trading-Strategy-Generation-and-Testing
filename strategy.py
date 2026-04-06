#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 12-hour trend filter and 1-day volume confirmation.
# Elder Ray measures bullish/bearish power relative to EMA13 to identify institutional buying/selling.
# 12-hour EMA50 trend filter ensures trades align with intermediate trend.
# 1-day volume surge (>1.5x 20-period average) confirms institutional participation.
# Designed for 6h timeframe to target 50-150 trades over 4 years with controlled frequency.
# Works in bull markets via bull power + uptrend, in bear via bear power + downtrend.

name = "6h_elderray12h_trend1d_vol_v1"
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
    
    # EMA13 for Elder Ray calculation (primary timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12-hour EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 1-day volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (EMA13 needs ~26 bars for stability)
    start = 26
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current 1-day volume > 1.5x 20-day average
        # Need to map current 6h bar to appropriate 1-day volume
        # Use the most recent available 1-day volume data
        vol_filter = volume_1d[-1] > vol_ma_1d_aligned[i] * 1.5 if i < len(vol_ma_1d_aligned) and not np.isnan(vol_ma_1d_aligned[i]) else False
        
        # Trend condition: price above/below 12-hour EMA50
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]
        
        # Check exits and stoploss (2.5 * ATR approximation using high-low range)
        if position == 1:  # long position
            # Exit: bear power turns negative or stoploss
            if (bear_power[i] > 0 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power turns positive or stoploss
            if (bull_power[i] < 0 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if vol_filter:
                # Long: bull power positive AND uptrend
                if bull_power[i] > 0 and uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bear power negative AND downtrend
                elif bear_power[i] < 0 and downtrend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals