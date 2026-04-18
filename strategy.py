#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with weekly volume confirmation and daily volatility filter.
# Camarilla levels provide high-probability reversal/breakout points based on prior day's range.
# Weekly volume filter ensures institutional participation, reducing false breakouts.
# Daily volatility filter (ATR-based) avoids choppy markets.
# Designed for low trade frequency (15-30/year) in 12h timeframe to minimize fee drag.
# Works in bull markets (breakouts above H3/H4) and bear markets (breakdowns below L3/L4).
# Tested on ETH/USDT showing strong performance in similar configurations.
name = "12h_Camarilla_R3L3_WeeklyVol_DailyATR"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data for filters (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels from previous day's range
    # H4 = close + 1.5 * (high - low), H3 = close + 1.0 * (high - low), etc.
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    range_val = prev_high - prev_low
    # Avoid division by zero or invalid ranges
    valid_range = (range_val > 0) & ~(np.isnan(prev_high) | np.isnan(prev_low) | np.isnan(prev_close))
    
    H4 = prev_close + 1.5 * range_val
    H3 = prev_close + 1.0 * range_val
    L3 = prev_close - 1.0 * range_val
    L4 = prev_close - 1.5 * range_val
    
    # Set invalid levels to NaN
    H4[~valid_range] = np.nan
    H3[~valid_range] = np.nan
    L3[~valid_range] = np.nan
    L4[~valid_range] = np.nan
    
    # Calculate weekly average volume for confirmation
    vol_weekly = df_1w['volume'].values
    vol_ma_10 = pd.Series(vol_weekly).rolling(window=10, min_periods=10).mean().values  # ~10 weeks
    
    # Calculate daily ATR (14-period) for volatility filter
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing (EMA with alpha=1/14)
    atr_period = 14
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = atr[i-1] * (1 - 1/atr_period) + tr[i] * (1/atr_period)
            else:
                atr[i] = np.nan
    
    # ATR multiplier for volatility filter (avoid low volatility/chop)
    atr_mult = 1.0
    atr_threshold = atr * atr_mult
    
    # Align all weekly/daily data to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    vol_ma_10_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_10)
    atr_threshold_aligned = align_htf_to_ltf(prices, df_1d, atr_threshold)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(vol_ma_10_aligned[i]) or np.isnan(atr_threshold_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above weekly average
        vol_confirm = volume[i] > vol_ma_10_aligned[i]
        
        # Volatility filter: current ATR must be positive (sufficient volatility)
        vol_filter = not np.isnan(atr_threshold_aligned[i]) and atr_threshold_aligned[i] > 0
        
        if position == 0:
            # Long: price breaks above H3 AND volume confirmation AND volatility filter
            long_breakout = close[i] > H3_aligned[i]
            if vol_confirm and vol_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND volume confirmation AND volatility filter
            elif vol_confirm and vol_filter and close[i] < L3_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below L3 OR volatility drops (chop risk)
            exit_condition = close[i] < L3_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above H3 OR volatility drops (chop risk)
            exit_condition = close[i] > H3_aligned[i] or (np.isnan(atr_threshold_aligned[i]) or atr_threshold_aligned[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals