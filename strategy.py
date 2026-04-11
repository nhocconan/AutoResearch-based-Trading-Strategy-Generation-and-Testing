#!/usr/bin/env python3
"""
1d_1w_camarilla_volatility_filter_v1
Strategy: Daily Camarilla pivot breakout with volatility filter and weekly trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses daily Camarilla pivot levels (H4/L4) for breakout entries, filtered by weekly trend (price above/below weekly EMA20) and volatility expansion (current ATR > 1.5x ATR(20)). Designed to capture strong breakouts in trending markets while avoiding false breakouts in low volatility. Weekly trend filter ensures alignment with higher timeframe momentum, volatility filter ensures sufficient momentum for breakout to succeed. Target: 20-50 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_volatility_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily ATR(20) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Camarilla levels (based on previous day's OHLC)
    # We need previous day's high, low, close for each day
    # Shift by 1 to get previous day's values
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    # Set first day's previous values to NaN (no prior day)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Camarilla levels: H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    camarilla_H4 = close_prev + 1.1 * (high_prev - low_prev) / 2
    camarilla_L4 = close_prev - 1.1 * (high_prev - low_prev) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(camarilla_H4[i]) or np.isnan(camarilla_L4[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Volatility filter: current ATR > 1.5 * ATR(20)
        vol_filter = atr[i] > (1.5 * atr[i-20] if i >= 20 and not np.isnan(atr[i-20]) else atr[i])
        
        # Weekly trend filter
        uptrend_1w = price_close > ema_20_1w_aligned[i]
        downtrend_1w = price_close < ema_20_1w_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > camarilla_H4[i]
        breakout_down = price_close < camarilla_L4[i]
        
        # Long: upward breakout with volatility expansion in uptrend
        long_signal = breakout_up and vol_filter and uptrend_1w
        
        # Short: downward breakout with volatility expansion in downtrend
        short_signal = breakout_down and vol_filter and downtrend_1w
        
        # Exit when price returns to the opposite Camarilla level
        exit_long = position == 1 and price_close < camarilla_L4[i]
        exit_short = position == -1 and price_close > camarilla_H4[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals