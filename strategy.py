#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1w ATR volatility filter
    # Works in bull/bear by capturing breakouts only when volatility is expanding
    # (avoids false breakouts in chop). 1w ATR filter ensures we only trade
    # during high volatility regimes, reducing whipsaws in ranging markets.
    # Target: 12-37 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for context (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # 1w ATR(14) for volatility filter and stop
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        atr_1w[i] = np.mean(tr[i-14:i+1])
    
    # Align 1w ATR to 12h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # 12h Donchian(20) for breakout signals
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 0.4 * its 20-period average
        atr_ma_20_1w = np.full(len(df_1w), np.nan)
        for j in range(33, len(df_1w)):
            if not np.isnan(np.mean(atr_1w[j-19:j+1])):
                atr_ma_20_1w[j] = np.mean(atr_1w[j-19:j+1])
        atr_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_20_1w)
        vol_filter = (not np.isnan(atr_ma_20_1w_aligned[i]) and 
                     atr_1w_aligned[i] > 0.4 * atr_ma_20_1w_aligned[i])
        
        # Breakout conditions
        breakout_long = close[i] > donch_high[i]
        breakout_short = close[i] < donch_low[i]
        
        # Entry conditions
        long_entry = breakout_long and vol_filter
        short_entry = breakout_short and vol_filter
        
        # Exit conditions: opposite breakout or volatility collapse
        long_exit = (close[i] < donch_low[i]) or (not vol_filter)
        short_exit = (close[i] > donch_high[i]) or (not vol_filter)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_donchian_breakout_vol_filter_v1"
timeframe = "12h"
leverage = 1.0