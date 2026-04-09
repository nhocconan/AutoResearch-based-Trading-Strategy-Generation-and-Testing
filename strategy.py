#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot long/short with 4h trend filter and session filter (08-20 UTC)
# Uses 4h EMA50 for trend direction and Camarilla H3/L3 levels for mean-reversion entries
# Session filter reduces noise trades during low-volume hours
# Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing 0.20
# Works in bull/bear: 4h trend filter avoids counter-trend trades, Camarilla provides precise reversal levels

name = "1h_4h_camarilla_trend_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend direction
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla H3, L3, H4, L4 levels
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(n):
        # Find previous day's index (24h ago in 1d data)
        if i < 24:  # Not enough 1h data for previous day
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
            continue
            
        # Get 1d index for previous day (approximately i//24 - 1)
        idx_1d = i // 24
        if idx_1d < 1:
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
            continue
            
        # Previous day's OHLC (yesterday's data)
        ph = high_1d[idx_1d - 1]
        pl = low_1d[idx_1d - 1]
        pc = close_1d[idx_1d - 1]
        
        if np.isnan(ph) or np.isnan(pl) or np.isnan(pc):
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
            continue
            
        range_val = ph - pl
        camarilla_h3[i] = pc + range_val * 1.1 / 4
        camarilla_l3[i] = pc - range_val * 1.1 / 4
        camarilla_h4[i] = pc + range_val * 1.1 / 2
        camarilla_l4[i] = pc - range_val * 1.1 / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(camarilla_h4[i]) or 
            np.isnan(camarilla_l4[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 OR 4h EMA50 trend turns bearish
            if close[i] < camarilla_l3[i] or close[i] < ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 OR 4h EMA50 trend turns bullish
            if close[i] > camarilla_h3[i] or close[i] > ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic with Camarilla levels and 4h trend filter
            # Long: price crosses above Camarilla H3 AND 4h trend is bullish
            if close[i] > camarilla_h3[i] and close[i] > ema_50_4h_aligned[i]:
                position = 1
                signals[i] = 0.20
            # Short: price crosses below Camarilla L3 AND 4h trend is bearish
            elif close[i] < camarilla_l3[i] and close[i] < ema_50_4h_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals