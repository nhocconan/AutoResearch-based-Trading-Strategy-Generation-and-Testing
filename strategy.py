#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h Donchian channel breakout with volume confirmation and 1d trend filter
# Uses 4h Donchian(20) breakout for signal direction, 1h for entry timing precision
# Volume spike (>1.5x 20-bar avg) confirms breakout strength
# 1d EMA(50) filter ensures alignment with higher timeframe trend
# Session filter (08-20 UTC) reduces noise trades
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
# Works in bull markets via upward breakouts and in bear markets via downward breakouts

name = "1h_Donchian20_Breakout_VolumeSpike_1dEMA50_Trend_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channel (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper = rolling max of high, lower = rolling min of low
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe (only available after 4h bar closes)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50)  # Donchian(20), EMA(50)
    
    for i in range(start_idx, n):
        # Skip if outside trading session or any required data is NaN
        if not in_session[i] or \
           np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        curr_upper = upper_4h_aligned[i]
        curr_lower = lower_4h_aligned[i]
        curr_ema = ema_50_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 4h Donchian upper, above 1d EMA50, volume spike
            if price > curr_upper and price > curr_ema and vol_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short entry: price breaks below 4h Donchian lower, below 1d EMA50, volume spike
            elif price < curr_lower and price < curr_ema and vol_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or Donchian lower break
            # ATR-based stoploss: 2.0 * ATR below entry (using 1h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            if price < stop_loss or price < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit on stoploss or Donchian upper break
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            if price > stop_loss or price > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals