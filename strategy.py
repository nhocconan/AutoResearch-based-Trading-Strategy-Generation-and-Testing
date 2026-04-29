#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Camarilla levels provide precise intraday support/resistance; EMA50 defines 1d trend direction;
# volume confirms breakout strength. Works in both bull/bear markets by trading with the
# 1d trend only when price breaks key Camarilla levels. Target: 12-30 trades/year (50-120 total).

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    # Camarilla equations:
    # H4 = Close + 1.5*(High - Low)
    # H3 = Close + 1.25*(High - Low)
    # H2 = Close + 1.166*(High - Low)
    # H1 = Close + 0.833*(High - Low)
    # L1 = Close - 0.833*(High - Low)
    # L2 = Close - 1.166*(High - Low)
    # L3 = Close - 1.25*(High - Low)
    # L4 = Close - 1.5*(High - Low)
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    daily_range = prev_day_high - prev_day_low
    h4 = prev_day_close + 1.5 * daily_range
    h3 = prev_day_close + 1.25 * daily_range
    h2 = prev_day_close + 1.166 * daily_range
    h1 = prev_day_close + 0.833 * daily_range
    l1 = prev_day_close - 0.833 * daily_range
    l2 = prev_day_close - 1.166 * daily_range
    l3 = prev_day_close - 1.25 * daily_range
    l4 = prev_day_close - 1.5 * daily_range
    
    # Align Camarilla levels to 12h timeframe (completed daily bar only)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, l1)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for 1d EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        curr_h4 = h4_aligned[i]
        curr_h3 = h3_aligned[i]
        curr_h2 = h2_aligned[i]
        curr_h1 = h1_aligned[i]
        curr_l1 = l1_aligned[i]
        curr_l2 = l2_aligned[i]
        curr_l3 = l3_aligned[i]
        curr_l4 = l4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long when: price breaks above H3 (strong resistance) + above 1d EMA50 + volume confirmation
            if curr_close > curr_h3 and curr_close > curr_ema_50 and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short when: price breaks below L3 (strong support) + below 1d EMA50 + volume confirmation
            elif curr_close < curr_l3 and curr_close < curr_ema_50 and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price falls below H1 (3/4 level) OR breaks below 1d EMA50
            if curr_close < curr_h1 or curr_close < curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price rises above L1 (3/4 level) OR breaks above 1d EMA50
            if curr_close > curr_l1 or curr_close > curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals