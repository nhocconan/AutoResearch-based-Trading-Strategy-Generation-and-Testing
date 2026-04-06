#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Reversal with Volume Confirmation
# Uses weekly pivot points (PP, R1-4, S1-4) for mean reversion in ranging markets.
# Long when price touches S1/S2 with bullish rejection and volume spike.
# Short when price touches R1/R2 with bearish rejection and volume spike.
# Trend filter: price must be between weekly R2 and S2 to avoid strong trends.
# Target: 80-150 total trades over 4 years with controlled risk.

name = "6h_weekly_pivot_reversion_v1"
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
    
    # Weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot point calculation: PP = (H + L + C)/3
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # Resistance and support levels
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    r4 = pp + 3 * (weekly_high - weekly_low)
    s4 = pp - 3 * (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # Volume average (24-period for 6h = ~6 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(24, n):  # Start after volume MA warmup
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches midpoint or opposite support level
            elif close[i] >= pp_aligned[i] or close[i] >= s2_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches midpoint or opposite resistance level
            elif close[i] <= pp_aligned[i] or close[i] <= r2_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion entries at support/resistance with volume confirmation
            # Long: price at S1/S2 with bullish rejection (close > open) and volume spike
            if ((abs(close[i] - s1_aligned[i]) < 0.1 * atr[i] or abs(close[i] - s2_aligned[i]) < 0.1 * atr[i]) and
                close[i] > prices['open'].iloc[i] and
                volume[i] > 2.0 * volume_ma[i] and
                close[i] < pp_aligned[i]):  # Ensure we're in ranging market (below pivot)
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price at R1/R2 with bearish rejection (close < open) and volume spike
            elif ((abs(close[i] - r1_aligned[i]) < 0.1 * atr[i] or abs(close[i] - r2_aligned[i]) < 0.1 * atr[i]) and
                  close[i] < prices['open'].iloc[i] and
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] > pp_aligned[i]):  # Ensure we're in ranging market (above pivot)
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals