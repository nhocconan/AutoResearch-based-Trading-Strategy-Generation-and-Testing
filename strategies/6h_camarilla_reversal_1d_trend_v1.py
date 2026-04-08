#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 1d trend filter
# Long when price crosses above S3 (Camarilla support) and 1d close > 1d EMA50 (uptrend)
# Short when price crosses below R3 (Camarilla resistance) and 1d close < 1d EMA50 (downtrend)
# Exit when price crosses opposite Camarilla level (S4 for long, R4 for short)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses Camarilla levels from 6h OHLC and trend filter from 1d EMA
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_camarilla_reversal_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below S4 (Camarilla support level 4)
            elif close[i] < low[i-3] - 1.128 * (high[i-3] - low[i-3]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above R4 (Camarilla resistance level 4)
            elif close[i] > high[i-3] + 1.128 * (high[i-3] - low[i-3]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Calculate Camarilla levels from previous bar (6h)
            # Using high, low, close from 3 bars ago (previous completed 6h bar)
            prev_high = high[i-3]
            prev_low = low[i-3]
            prev_close = close[i-3]
            range_val = prev_high - prev_low
            
            # Camarilla levels
            s3 = prev_close - 1.128 * range_val / 6
            r3 = prev_close + 1.128 * range_val / 6
            s4 = prev_close - 1.500 * range_val / 2
            r4 = prev_close + 1.500 * range_val / 2
            
            # Trend filter: 1d EMA50
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Long: price crosses above S3 (support) in uptrend
            if close[i] > s3 and close[i-1] <= s3 and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price crosses below R3 (resistance) in downtrend
            elif close[i] < r3 and close[i-1] >= r3 and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals