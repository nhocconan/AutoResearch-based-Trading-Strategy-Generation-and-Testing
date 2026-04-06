#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price action around weekly pivot levels with volume confirmation.
# Uses weekly pivot (from Monday open) as key support/resistance.
# Goes long when price bounces off weekly pivot support with volume > 1.5x average.
# Goes short when price is rejected at weekly pivot resistance with volume > 1.5x average.
# Filters trades by 12h EMA trend to avoid counter-trend trades in strong trends.
# Target: 60-180 total trades over 4 years (15-45/year) with controlled risk.

name = "6h_weeklypivot_vol_trend_v1"
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
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation (weekly pivot from Monday open)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points using prior week's data
    # Using standard pivot: P = (H + L + C) / 3
    # Support/resistance levels: R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Calculate pivot and levels
    pivot = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    r3 = high_w + 2 * (pivot - low_w)
    s3 = low_w - 2 * (high_w - pivot)
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma  # Volume above average
    vol_strong = volume > (vol_ma * 1.5)  # Strong volume for confirmations
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below S2 or 12h EMA turns down
            elif close[i] < s2_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above R2 or 12h EMA turns up
            elif close[i] > r2_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if vol_strong[i]:
                # Bounce long at S1/S2: price finds support with strong volume
                if (close[i] <= s2_aligned[i] * 1.002 and close[i] >= s2_aligned[i] * 0.998) or \
                   (close[i] <= s1_aligned[i] * 1.002 and close[i] >= s1_aligned[i] * 0.998):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Rejection short at R1/R2: price hits resistance with strong volume
                elif (close[i] >= r1_aligned[i] * 0.998 and close[i] <= r1_aligned[i] * 1.002) or \
                     (close[i] >= r2_aligned[i] * 0.998 and close[i] <= r2_aligned[i] * 1.002):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            # Counter-trend entries at S3/R3 with volume (weaker signals)
            elif vol_filter[i]:
                # Long at S3: extreme oversold bounce
                if close[i] <= s3_aligned[i] * 1.001 and close[i] >= s3_aligned[i] * 0.999:
                    signals[i] = 0.15
                    position = 1
                    entry_price = close[i]
                # Short at R3: extreme overbought rejection
                elif close[i] >= r3_aligned[i] * 0.999 and close[i] <= r3_aligned[i] * 1.001:
                    signals[i] = -0.15
                    position = -1
                    entry_price = close[i]
    
    return signals