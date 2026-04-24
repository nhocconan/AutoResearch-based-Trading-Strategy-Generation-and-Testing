#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and 1d volume spike.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA50 trend direction, 1d for volume average confirmation.
- Camarilla levels: calculated from previous day's OHLC (H3/L3 for breakout, H4/L4 for stops).
- Entry: Long when price breaks above H3 with 4h EMA50 uptrend AND volume > 1.5 * 1d average volume.
         Short when price breaks below L3 with 4h EMA50 downtrend AND volume > 1.5 * 1d average volume.
- Exit: Opposite Camarilla breakout (L3 for longs, H3 for shorts) or time-based exit after 24h.
- Signal size: 0.20 discrete to minimize fee drag.
- Camarilla pivots identify intraday support/resistance levels that work in ranging markets.
- Volume confirmation ensures breakout legitimacy.
- 4h EMA50 filter aligns with medium-term trend to avoid counter-trend whipsaws.
- Works in both bull and bear markets as it captures volatility expansion from key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d average volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Need to group by day to get previous day's OHLC
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    # Arrays to store Camarilla levels for each bar
    h3 = np.full(n, np.nan)
    l3 = np.full(n, np.nan)
    h4 = np.full(n, np.nan)
    l4 = np.full(n, np.nan)
    
    # Calculate for each day
    for i in range(1, len(unique_dates)):
        prev_date = unique_dates[i-1]
        curr_date = unique_dates[i]
        
        # Get previous day's OHLC
        prev_day_mask = (dates == prev_date)
        if not np.any(prev_day_mask):
            continue
            
        prev_high = np.max(high[prev_day_mask])
        prev_low = np.min(low[prev_day_mask])
        prev_close = close[prev_day_mask][-1]  # Last close of previous day
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        h3_val = prev_close + range_val * 1.1 / 4
        l3_val = prev_close - range_val * 1.1 / 4
        h4_val = prev_close + range_val * 1.1 / 2
        l4_val = prev_close - range_val * 1.1 / 2
        
        # Apply to current day's bars
        curr_day_mask = (dates == curr_date)
        h3[curr_day_mask] = h3_val
        l3[curr_day_mask] = l3_val
        h4[curr_day_mask] = h4_val
        l4[curr_day_mask] = l4_val
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = np.full(n, -1)  # Track entry bar for time-based exit
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(h3[i]) or np.isnan(l3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_bar[i] = -1
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Track entry bar for time-based exit
        if position != 0 and entry_bar[i-1] != -1:
            entry_bar[i] = entry_bar[i-1]
        elif position != 0:
            entry_bar[i] = i
        
        # Time-based exit: close position after 24 hours (24 bars on 1h)
        if position != 0 and entry_bar[i] != -1 and (i - entry_bar[i]) >= 24:
            signals[i] = 0.0
            position = 0
            entry_bar[i] = -1
            continue
        
        # Exit conditions: opposite Camarilla breakout
        if position != 0:
            # Exit long: price breaks below L3
            if position == 1:
                if curr_low <= l3[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_bar[i] = -1
                    continue
            # Exit short: price breaks above H3
            elif position == -1:
                if curr_high >= h3[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_bar[i] = -1
                    continue
        
        # Entry conditions: Camarilla breakout with 4h trend filter and volume confirmation
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_high >= h3[i] and prev_close < h3[i-1]
            breakout_down = curr_low <= l3[i] and prev_close > l3[i-1]
            
            # 4h trend filter: EMA50 direction
            uptrend = ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]
            downtrend = ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
                entry_bar[i] = i
            elif breakout_down and downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
                entry_bar[i] = i
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50Trend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0