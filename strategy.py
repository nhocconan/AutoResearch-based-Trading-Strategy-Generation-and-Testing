#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ATR breakout with weekly trend filter and volume confirmation.
# Uses 1d ATR for breakout detection and 1w EMA for trend direction.
# Breakouts above/below ATR(14) from open + volume confirmation.
# Trend filter: only long when price > weekly EMA(50), only short when price < weekly EMA(50).
# Volatility filter: only trade when ATR(14) > 20-period average ATR (avoid chop).
# Designed to work in both bull (breakout continuation) and bear (breakout reversal) markets.
# Target: 50-100 total trades over 4 years (12-25/year) with controlled risk.

name = "1d_atr_breakout_weekly_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d ATR calculation
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]  # First value
    low_close[0] = high_low[0]   # First value
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend direction
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_50)  # Already shifted by 1 week
    
    # Volume moving average for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(atr_ma[i]) or 
            np.isnan(ema_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Filters
        vol_filter = volume[i] > vol_ma[i]  # Volume above average
        vol_filter_2 = atr[i] > atr_ma[i]   # Volatility above average (avoid chop)
        
        if position == 1:  # long position
            # Exit: price closes below open - ATR (breakdown) or trend reversal
            if close[i] < open_price[i] - atr[i] or close[i] < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above open + ATR (breakout) or trend reversal
            if close[i] > open_price[i] + atr[i] or close[i] > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with filters
            if vol_filter and vol_filter_2:
                # Long breakout: price breaks above open + ATR with uptrend
                if close[i] > open_price[i] + atr[i] and close[i] > ema_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: price breaks below open - ATR with downtrend
                elif close[i] < open_price[i] - atr[i] and close[i] < ema_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals