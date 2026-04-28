#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for HTF context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Weekly high-low for range calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly ATR(14) for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14_w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Weekly EMA(21) for trend
    ema_21_w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly indicators to daily timeframe
    ema_21_w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_w)
    atr_14_w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_w)
    
    # Daily ATR(10) for position sizing and stop
    tr1_d = high - low
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    atr_10_d = pd.Series(tr_d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Weekly range position: where price is within weekly range
    weekly_range = high_1w - low_1w
    # Avoid division by zero
    weekly_range_safe = np.where(weekly_range == 0, 1, weekly_range)
    range_position = (close - low_1w) / weekly_range_safe  # 0 at low, 1 at high
    
    # Align range position to daily
    range_position_aligned = align_htf_to_ltf(prices, df_1w, range_position)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_w_aligned[i]) or np.isnan(atr_14_w_aligned[i]) or 
            np.isnan(range_position_aligned[i]) or np.isnan(atr_10_d[i])):
            signals[i] = 0.0
            continue
        
        # Range filter: avoid extremes (mean reversion in middle 60% of range)
        in_middle_range = (range_position_aligned[i] > 0.2) & (range_position_aligned[i] < 0.8)
        
        # Trend filter: weekly EMA slope
        if i >= 2:
            ema_slope = ema_21_w_aligned[i] - ema_21_w_aligned[i-2]
            trend_up = ema_slope > 0
            trend_down = ema_slope < 0
        else:
            trend_up = trend_down = False
        
        # Volatility filter: avoid extremely high volatility weeks
        vol_ma = pd.Series(atr_14_w_aligned).ewm(span=20, adjust=False, min_periods=20).mean().values
        vol_normal = atr_14_w_aligned[i] < 2.0 * vol_ma[i]
        
        # Entry conditions: mean reversion in trending market with normal volatility
        long_entry = in_middle_range and trend_down and vol_normal  # Sell the rally in uptrend
        short_entry = in_middle_range and trend_up and vol_normal   # Buy the dip in downtrend
        
        # Exit conditions: return to extreme or volatility spike
        long_exit = (range_position_aligned[i] <= 0.2) or (range_position_aligned[i] >= 0.8) or (atr_14_w_aligned[i] > 2.5 * vol_ma[i])
        short_exit = (range_position_aligned[i] <= 0.2) or (range_position_aligned[i] >= 0.8) or (atr_14_w_aligned[i] > 2.5 * vol_ma[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyRange_MeanReversion_TrendFilter"
timeframe = "1d"
leverage = 1.0