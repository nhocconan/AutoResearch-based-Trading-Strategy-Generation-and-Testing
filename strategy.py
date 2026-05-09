# 6H_GapFill_1dTrend_Volume
# Hypothesis: 6h gap fill strategy with 1d trend filter and volume confirmation.
# Gaps often fill due to mean reversion, especially in choppy markets.
# In strong trends, we trade pullbacks to the gap area in trend direction.
# Works in both bull (buy dips) and bear (sell rallies) markets.
# Target: 15-35 trades/year per symbol.

name = "6H_GapFill_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and gap detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA21 for trend filter
    ema_21_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 21:
        ema_21_1d[20] = np.mean(close_1d[0:21])
        for i in range(21, len(close_1d)):
            ema_21_1d[i] = (ema_21_1d[i-1] * 20 + close_1d[i]) / 21
    
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Daily ATR14 for gap threshold
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], tr_1d])
    
    atr_14_1d = np.full_like(tr_1d, np.nan)
    if len(tr_1d) >= 14:
        atr_14_1d[13] = np.mean(tr_1d[0:14])
        for i in range(14, len(tr_1d)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr_1d[i]) / 14
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 6h volume ratio (current vs 6-period average = 1 day)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 6:
        vol_ma[5] = np.mean(volume[0:6])
        for i in range(6, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 5 + volume[i]) / 6
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(6, 21)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_21_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Detect gap: previous 6h bar's close vs current bar's open
            prev_close = close[i-1]
            curr_open = prices['open'].iloc[i]
            gap_size = abs(curr_open - prev_close)
            
            # Only trade if gap is significant (> 0.5 * daily ATR)
            if gap_size > 0.5 * atr_14_1d_aligned[i]:
                # Gap down: potential long (price likely to fill gap upward)
                if curr_open < prev_close:
                    # In uptrend or ranging: buy the gap fill
                    if close[i-1] >= ema_21_1d_aligned[i-1] or volume_ratio[i] > 1.5:
                        signals[i] = 0.25
                        position = 1
                        bars_since_entry = 0
                # Gap up: potential short (price likely to fill gap downward)
                elif curr_open > prev_close:
                    # In downtrend or ranging: sell the gap fill
                    if close[i-1] <= ema_21_1d_aligned[i-1] or volume_ratio[i] > 1.5:
                        signals[i] = -0.25
                        position = -1
                        bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit conditions:
                # 1. Gap filled (price returned to previous close)
                # 2. Strong adverse move (stop)
                prev_close = close[i-1]
                if close[i] >= prev_close or close[i] < prev_close - 1.5 * atr_14_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit conditions:
                # 1. Gap filled (price returned to previous close)
                # 2. Strong adverse move (stop)
                prev_close = close[i-1]
                if close[i] <= prev_close or close[i] > prev_close + 1.5 * atr_14_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals