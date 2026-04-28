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
    
    # Get daily data for ATR and momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily ATR(14) for volatility normalization
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = abs(df_1d['low'] - df_1d['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Daily RSI(14) for momentum
    delta = df_1d['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi14 = (100 - (100 / (1 + rs))).values
    rsi14_aligned = align_htf_to_ltf(prices, df_1d, rsi14)
    
    # Get 4h data for price structure
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h High-Low range for volatility-adjusted breakout
    hl_range = df_4h['high'] - df_4h['low']
    hl_ma = hl_range.rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr14_aligned[i]) or np.isnan(rsi14_aligned[i]) or 
            np.isnan(hl_ma[i])):
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
        
        # Volatility filter: current 4h range above average
        current_range = high[i] - low[i]
        vol_filter = current_range > (0.5 * hl_ma[i])
        
        # Momentum filter: RSI not extreme
        rsi = rsi14_aligned[i]
        mom_filter = (rsi > 30) and (rsi < 70)
        
        # Volatility-adjusted breakout levels
        atr = atr14_aligned[i]
        upper_break = hl_ma[i] + (0.5 * atr)
        lower_break = hl_ma[i] - (0.5 * atr)
        
        # Entry conditions
        long_entry = (current_range > upper_break) and vol_filter and mom_filter
        short_entry = (current_range > lower_break) and vol_filter and mom_filter
        
        # Exit conditions: volatility contraction
        vol_exit = current_range < (0.3 * hl_ma[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif vol_exit:
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

name = "4h_VolMom_Breakout"
timeframe = "4h"
leverage = 1.0