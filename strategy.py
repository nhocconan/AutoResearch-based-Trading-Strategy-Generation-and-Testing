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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from weekly EMA
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Entry conditions: breakout of recent range with volume confirmation
        lookback = 10
        if i >= lookback:
            recent_high = np.nanmax(high[i-lookback:i])
            recent_low = np.nanmin(low[i-lookback:i])
            
            # Volume filter: current volume above recent average
            vol_ma = np.nanmean(volume[max(0,i-5):i]) if i >= 5 else volume[i]
            volume_filter = volume[i] > 1.3 * vol_ma
            
            long_breakout = close[i] > recent_high
            short_breakout = close[i] < recent_low
            
            long_entry = uptrend and long_breakout and volume_filter
            short_entry = downtrend and short_breakout and volume_filter
        else:
            long_entry = False
            short_entry = False
        
        # Exit conditions: ATR-based trailing stop
        if position == 1:
            # Trail stop: exit if price drops 2.5*ATR from highest high since entry
            lookback_stop = min(20, i+1)
            recent_high = np.nanmax(high[i-lookback_stop:i+1])
            exit_condition = close[i] < recent_high - 2.5 * atr_1d_aligned[i]
        elif position == -1:
            # Trail stop: exit if price rises 2.5*ATR from lowest low since entry
            lookback_stop = min(20, i+1)
            recent_low = np.nanmin(low[i-lookback_stop:i+1])
            exit_condition = close[i] > recent_low + 2.5 * atr_1d_aligned[i]
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
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

name = "1d_1wEMA34_Breakout_Volume_ATRStop"
timeframe = "1d"
leverage = 1.0