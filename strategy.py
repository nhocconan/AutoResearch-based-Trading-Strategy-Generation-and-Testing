#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with weekly trend filter and volume confirmation
# Uses 1d Camarilla levels (H3/L3, H4/L4) for breakout entries, 1w EMA50 for trend direction,
# and volume spike for confirmation. Works in bull/bear by only taking breakouts in
# direction of weekly trend. Target: 15-25 trades/year (60-100 total).

name = "1d_Camarilla_H3L3_Breakout_WeeklyTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Calculate 1d Camarilla pivot points (using previous day's OHLC)
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.1 * (High - Low)
    # L3 = Close - 1.1 * (High - Low)
    prev_day_high = prices['high'].shift(1).values
    prev_day_low = prices['low'].shift(1).values
    prev_day_close = prices['close'].shift(1).values
    
    daily_range = prev_day_high - prev_day_low
    h4 = prev_day_close + 1.5 * daily_range
    l4 = prev_day_close - 1.5 * daily_range
    h3 = prev_day_close + 1.1 * daily_range
    l3 = prev_day_close - 1.1 * daily_range
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for weekly EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(weekly_ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_weekly_ema50 = weekly_ema50_aligned[i]
        curr_h4 = h4[i]
        curr_l4 = l4[i]
        curr_h3 = h3[i]
        curr_l3 = l3[i]
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price closes above H4 in uptrend (close > weekly EMA50)
            if curr_close > curr_h4 and curr_close > curr_weekly_ema50 and curr_volume_confirm:
                signals[i] = 0.30
                position = 1
            # Short breakdown: price closes below L4 in downtrend (close < weekly EMA50)
            elif curr_close < curr_l4 and curr_close < curr_weekly_ema50 and curr_volume_confirm:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:  # Long position - exit when price closes below H3
            if curr_close < curr_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position - exit when price closes above L3
            if curr_close > curr_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals