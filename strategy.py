#!/usr/bin/env python3
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
    
    # Get 1d data once for context (daily EMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for long-term trend
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200 = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get weekly data for major trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily ATR for volatility filter and position sizing
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14 = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily volume average for confirmation
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20 = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Hour filter: 8-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filters: price above both daily EMA200 and weekly EMA50
        trend_up = close[i] > ema_200[i] and close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_200[i] and close[i] < ema_50_1w_aligned[i]
        
        # Volatility filter: avoid extremely high volatility days
        vol_filter = atr_14[i] < 2.5 * np.nanmedian(atr_14[max(0, i-50):i+1])
        
        # Volume filter: above average daily volume
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Entry conditions require all filters
        long_entry = trend_up and vol_filter and vol_confirm
        short_entry = trend_down and vol_filter and vol_confirm
        
        # Exit conditions: trend reversal or volatility spike
        long_exit = not trend_up or atr_14[i] > 3.0 * np.nanmedian(atr_14[max(0, i-20):i+1])
        short_exit = not trend_down or atr_14[i] > 3.0 * np.nanmedian(atr_14[max(0, i-20):i+1])
        
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

name = "1d_EMA200_1wEMA50_Vol_Filter"
timeframe = "1d"
leverage = 1.0