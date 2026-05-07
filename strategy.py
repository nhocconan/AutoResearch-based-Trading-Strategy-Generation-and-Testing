# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "6h_WeeklyPivot_DailyTrend_VolumeBreak"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend and pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly high/low for pivot reference (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    # Weekly pivot points (simplified: using prior week high/low/close)
    # We'll use the weekly high and low as reference levels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # 6h ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA and ATR
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or np.isnan(atr_14[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14[i] > 0.01 * close[i]  # At least 1% of price
        
        if position == 0:
            # Long: price above weekly high AND daily uptrend AND volatility filter
            if close[i] > weekly_high_aligned[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly low AND daily downtrend AND volatility filter
            elif close[i] < weekly_low_aligned[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below weekly low or trend changes
            if close[i] < weekly_low_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above weekly high or trend changes
            if close[i] > weekly_high_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s breakout of weekly pivot levels with daily trend filter
# - Weekly high/low act as significant support/resistance levels
# - Breakouts above weekly high or below weekly low indicate strong momentum
# - Daily EMA(34) ensures alignment with intermediate trend
# - Volatility filter (ATR > 1% of price) avoids choppy markets
# - Works in bull (buy weekly high breakouts in uptrend) and bear (sell weekly low breakdowns in downtrend)
# - Position size 0.25 targets 20-40 trades/year, avoiding excessive fees
# - Exit when price returns to weekly opposite level or trend reverses