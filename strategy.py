#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Camarilla levels calculated from prior 1d OHLC: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
- Long when price breaks above H3 with volume > 1.5 * volume MA(20) AND price > 1d EMA34 (uptrend)
- Short when price breaks below L3 with volume > 1.5 * volume MA(20) AND price < 1d EMA34 (downtrend)
- Exit when price returns to cam camarilla pivot (close) or opposite level (L3 for long, H3 for short)
- Designed to capture institutional breakouts with trend alignment and volume confirmation
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where we have at least 1 day of data for Camarilla calculation
    start_idx = 96  # 96 * 4h = 16 days? No: 4h bars per day = 6, so need 1 bar of 1d data -> 6 bars of 4h
                    # Actually: need prior 1d OHLC, so we start after we have first complete 1d bar
                    # 1d bar = 6 * 4h bars, so start_idx = 6 to have prior day's data
                    # But we also need EMA34 warmup, so max(6, 34*6) = 204 bars
    start_idx = max(6, 34 * 6)  # Need prior 1d OHLC and EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get prior 1d OHLC for Camarilla calculation
        # Prior 1d bar ended at index: (i // 6) * 6 - 6 (the 1d bar before current)
        # But we need to use completed 1d bar only
        # We'll use the align_htf_to_ltf approach for OHLC as well
        if i < 6:  # Need at least one prior 1d bar
            continue
            
        # Get the completed 1d bar that ended before current 4h bar
        # Number of completed 1d bars up to index i: i // 6
        # The last completed 1d bar ended at: (i // 6) * 6 - 1 (last 4h bar of that day)
        # But easier: use HTF data directly
        
        # Instead, we'll calculate Camarilla from the 1d HTF data we already have
        # We need to align the prior 1d bar's OHLC to each 4h bar
        # For simplicity, we'll use the prior completed 1d bar's values
        
        # Find index of prior completed 1d bar in 1d array
        # Current 4h bar index i corresponds to 1d bar index: i // 6
        # We want the 1d bar before that: (i // 6) - 1
        idx_1d = (i // 6) - 1
        if idx_1d < 0 or idx_1d >= len(df_1d):
            continue
            
        prior_high = df_1d['high'].iloc[idx_1d]
        prior_low = df_1d['low'].iloc[idx_1d]
        prior_close = df_1d['close'].iloc[idx_1d]
        
        # Calculate Camarilla levels
        range_ = prior_high - prior_low
        if range_ <= 0:
            continue
            
        H3 = prior_close + 1.1 * range_ / 4
        L3 = prior_close - 1.1 * range_ / 4
        pivot = prior_close
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * volume_ma[i]
        
        if position == 0:
            # Long: price breaks above H3 with volume AND uptrend
            if close[i] > H3 and vol_confirm and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with volume AND downtrend
            elif close[i] < L3 and vol_confirm and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot or breaks below L3
            if close[i] <= pivot or close[i] < L3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to pivot or breaks above H3
            if close[i] >= pivot or close[i] > H3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0