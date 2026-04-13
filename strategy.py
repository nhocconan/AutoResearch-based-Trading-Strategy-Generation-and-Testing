#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter
    # Long: price breaks above H3 level + volume > 1.3x 20-period 4h average + 1d close > 1d EMA50
    # Short: price breaks below L3 level + volume > 1.3x 20-period 4h average + 1d close < 1d EMA50
    # Uses discrete sizing (0.20) to minimize fee drag
    # Target: 60-150 total trades over 4 years = 15-37/year for 1h
    # Session filter: 08-20 UTC to reduce noise trades
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day)
    high_1d = df_4h['high'].values  # Using 4h high/low for intraday pivot calculation
    low_1d = df_4h['low'].values
    close_1d = df_4h['close'].values
    
    # For Camarilla calculation, we need daily OHLC - approximate using 4h data
    # In practice, we would use actual daily data, but for this implementation
    # we'll use 4h data resampled to daily equivalent (this is a simplification)
    # Better approach: get actual 1d data for pivot calculation
    high_1d_actual = df_1d['high'].values
    low_1d_actual = df_1d['low'].values
    close_1d_actual = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    pivot = (high_1d_actual + low_1d_actual + close_1d_actual) / 3.0
    hl_range = high_1d_actual - low_1d_actual
    h3 = pivot + 1.1 * hl_range
    l3 = pivot - 1.1 * hl_range
    
    # Calculate 4h volume average (20-period) for confirmation
    vol_avg_20_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d_actual).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR for 1h timeframe
    atr_1h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_1h[i] = tr  # Simple average for warmup
        else:
            atr_1h[i] = 0.93 * atr_1h[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(50, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
            continue
        
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or
            np.isnan(vol_avg_20_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_avg_20_4h_aligned[i]
        
        # Trend filter: 1h close above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions: price breaks Camarilla levels with volume and trend
        breakout_long = (close[i] > h3_aligned[i]) and volume_confirmed and uptrend
        breakout_short = (close[i] < l3_aligned[i]) and volume_confirmed and downtrend
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_1h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_1h[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "1h_4h_1d_camarilla_volume_trend_v1"
timeframe = "1h"
leverage = 1.0