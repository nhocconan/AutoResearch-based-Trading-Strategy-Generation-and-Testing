#!/usr/bin/env python3
"""
12h_1d_ari_consecutive_highs_lows_v1
Strategy: 12h Consecutive Higher Highs/Lows breakout with volume confirmation and 1d trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses 12h price breaking above 3 consecutive higher highs or below 3 consecutive lower lows confirmed by volume spike (>1.5x average volume) and filtered by 1d EMA50 trend direction. Designed to capture strong momentum moves while filtering out false breakouts in choppy markets. Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out). Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_ari_consecutive_highs_lows_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h Consecutive Higher Highs (3-bar)
    hh1 = high > np.roll(high, 1)
    hh2 = hh1 & np.roll(hh1, 1)
    consec_hh = hh2 & np.roll(hh2, 1)
    
    # 12h Consecutive Lower Lows (3-bar)
    ll1 = low < np.roll(low, 1)
    ll2 = ll1 & np.roll(ll1, 1)
    consec_ll = ll2 & np.roll(ll2, 1)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(vol_avg[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = consec_hh[i-1]  # Use previous bar's signal
        breakout_down = consec_ll[i-1]  # Use previous bar's signal
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend_1d
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend_1d
        
        # Exit when price returns to middle of recent range (simple mean of last 5 closes)
        if i >= 5:
            recent_mean = np.mean(close[i-5:i])
            exit_long = position == 1 and price_close < recent_mean
            exit_short = position == -1 and price_close > recent_mean
        else:
            exit_long = False
            exit_short = False
        
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