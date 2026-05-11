#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Combines 12h Camarilla R3/S3 breakouts with 1week trend structure and volume confirmation.
# Long when: 1) weekly structure is bullish (HH and HL), 2) price breaks above 12h Camarilla R3 level, 3) volume > 1.5x 20-period average.
# Short when: 1) weekly structure is bearish (LH and LL), 2) price breaks below 12h Camarilla S3 level, 3) volume > 1.5x 20-period average.
# Exits when price returns to the 12h EMA10 or structure breaks.
# Uses weekly trend to avoid counter-trend trades in strong trends, volume to confirm breakout validity.
# 12h timeframe limits trades to avoid fee drag (target: 12-37 trades/year).

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for structure (HH, HL, LH, LL)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w structure: HH/HL for uptrend, LH/LL for downtrend ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    # Higher High: today's high > yesterday's high
    hh = high_1w > np.roll(high_1w, 1)
    # Higher Low: today's low > yesterday's low
    hl = low_1w > np.roll(low_1w, 1)
    # Lower High: today's high < yesterday's high
    lh = high_1w < np.roll(high_1w, 1)
    # Lower Low: today's low < yesterday's low
    ll = low_1w < np.roll(low_1w, 1)
    # Uptrend: HH and HL
    uptrend = hh & hl
    # Downtrend: LH and LL
    downtrend = lh & ll
    # First bar: no previous week, set to False
    uptrend[0] = False
    downtrend[0] = False
    
    # --- 12h EMA10 for exit ---
    close_series = pd.Series(close)
    ema10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # --- 12h Camarilla levels (based on previous day's OHLC) ---
    # Note: Camarilla levels are typically calculated from daily OHLC, but we'll use 12h high/low/close as proxy
    # For true daily Camarilla, we would need to align daily data, but for simplicity and avoiding look-ahead,
    # we use the previous 12h bar's high, low, close to calculate levels for the current bar.
    # This assumes the 12h bar approximates the daily session for level calculation.
    ph = np.roll(high, 1)  # previous high
    pl = np.roll(low, 1)   # previous low
    pc = np.roll(close, 1) # previous close
    # First bar: no previous bar, set to 0 to avoid invalid levels
    ph[0] = ph[1] if len(ph) > 1 else 0
    pl[0] = pl[1] if len(pl) > 1 else 0
    pc[0] = pc[1] if len(pc) > 1 else 0
    
    # Calculate Camarilla levels
    R3 = pc + (ph - pl) * 1.1 / 2
    S3 = pc - (ph - pl) * 1.1 / 2
    
    # --- 12h volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align all 1w indicators to 12h timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1w, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1w, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA10, volume MA(20), and Camarilla (need 1 for previous bar)
    start_idx = max(10, 20, 1)  # EMA10, vol MA, and need at least 1 bar for Camarilla
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema10[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i]) or
            np.isnan(R3[i]) or
            np.isnan(S3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Structure from 1w
        is_uptrend = uptrend_aligned[i]
        is_downtrend = downtrend_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: weekly uptrend + volume spike + price above Camarilla R3
                if close[i] > R3[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: weekly downtrend + volume spike + price below Camarilla S3
                if close[i] < S3[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price returns to EMA10 OR structure breaks down
                if close[i] < ema10[i] or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to EMA10 OR structure breaks up
                if close[i] > ema10[i] or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals