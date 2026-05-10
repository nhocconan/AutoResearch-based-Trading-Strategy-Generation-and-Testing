#!/usr/bin/env python3
# 1d_KeltnerChannel_Breakout_WeeklyTrend_Volume
# Hypothesis: Breakout above/below Keltner Channel (20,2.0) on daily timeframe, filtered by weekly EMA50 trend and volume confirmation (>1.8x average).
# Uses ATR-based stoploss. Designed for 15-25 trades/year to avoid fee drag. Works in bull/bear via weekly trend filter.

name = "1d_KeltnerChannel_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for Keltner Channel and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Get weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Keltner Channel (20,2.0) on daily
    ema_middle = np.full(n, np.nan)
    kc_upper = np.full(n, np.nan)
    kc_lower = np.full(n, np.nan)
    for i in range(20, n):
        ema_middle[i] = np.nanmean(close[i-20:i])  # Simple MA for middle (can be EMA but SMA is fine)
        kc_upper[i] = ema_middle[i] + 2.0 * atr[i]
        kc_lower[i] = ema_middle[i] - 2.0 * atr[i]
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of weekly EMA50 trend
            if close[i] > ema_50_1w_aligned[i]:  # Uptrend
                # Long: Breakout above Keltner upper with volume confirmation
                if close[i] > kc_upper[i] and volume[i] > 1.8 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: Breakout below Keltner lower with volume confirmation
                if close[i] < kc_lower[i] and volume[i] > 1.8 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price closes below EMA middle or stoploss hit
            if close[i] < ema_middle[i] or (i > 0 and low[i] < kc_lower[i] - 1.5 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above EMA middle or stoploss hit
            if close[i] > ema_middle[i] or (i > 0 and high[i] > kc_upper[i] + 1.5 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals