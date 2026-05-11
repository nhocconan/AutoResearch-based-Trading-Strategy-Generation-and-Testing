#!/usr/bin/env python3
# 4h_ChaikinMoneyFlow_Breakout_1dTrend
# Hypothesis: Uses Chaikin Money Flow (CMF) to confirm institutional buying/selling pressure combined with daily trend structure.
# Long when: 1) daily structure bullish (HH & HL), 2) CMF(20) > 0.1 (strong buying pressure), 3) price breaks above 4h Donchian(20) upper band.
# Short when: 1) daily structure bearish (LH & LL), 2) CMF(20) < -0.1 (strong selling pressure), 3) price breaks below 4h Donchian(20) lower band.
# Exits when price returns to the 4h EMA(20) or daily structure breaks.
# CMF filters false breakouts by requiring volume-weighted money flow confirmation.
# Works in bull markets by buying strong uptrend pullbacks and in bear markets by selling rallies in downtrends.

name = "4h_ChaikinMoneyFlow_Breakout_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for structure (HH, HL, LH, LL)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d structure: HH/HL for uptrend, LH/LL for downtrend ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Higher High: today's high > yesterday's high
    hh = high_1d > np.roll(high_1d, 1)
    # Higher Low: today's low > yesterday's low
    hl = low_1d > np.roll(low_1d, 1)
    # Lower High: today's high < yesterday's high
    lh = high_1d < np.roll(high_1d, 1)
    # Lower Low: today's low < yesterday's low
    ll = low_1d < np.roll(low_1d, 1)
    # Uptrend: HH and HL
    uptrend = hh & hl
    # Downtrend: LH and LL
    downtrend = lh & ll
    # First bar: no previous day, set to False
    uptrend[0] = False
    downtrend[0] = False
    
    # --- 4h EMA(20) for exit ---
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # --- 4h Donchian(20) channels ---
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # --- 4h Chaikin Money Flow (20) ---
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high - low
    mfm = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0)
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    mfv_sum = np.full(n, np.nan)
    vol_sum = np.full(n, np.nan)
    for i in range(20, n):
        mfv_sum[i] = np.sum(mfv[i-20:i])
        vol_sum[i] = np.sum(volume[i-20:i])
    cmf = np.full(n, np.nan)
    for i in range(20, n):
        if vol_sum[i] != 0:
            cmf[i] = mfv_sum[i] / vol_sum[i]
        else:
            cmf[i] = 0.0
    
    # Align all 1d indicators to 4h timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Donchian(20), EMA20, CMF20
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(ema20[i]) or
            np.isnan(cmf[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Structure from 1d
        is_uptrend = uptrend_aligned[i]
        is_downtrend = downtrend_aligned[i]
        
        if position == 0:
            if is_uptrend and cmf[i] > 0.1:
                # Long: daily uptrend + buying pressure + price above Donchian high
                if close[i] > donch_high[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and cmf[i] < -0.1:
                # Short: daily downtrend + selling pressure + price below Donchian low
                if close[i] < donch_low[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price returns to EMA20 OR structure breaks down
                if close[i] < ema20[i] or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to EMA20 OR structure breaks up
                if close[i] > ema20[i] or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals