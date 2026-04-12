#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d MACD filter for mean reversion in range-bound markets
# Uses oversold/overbought conditions on 6s with trend filter from daily MACD
# Targets 20-40 trades/year by requiring confluence of momentum extreme and trend alignment
# Works in bull/bear by fading extremes only when higher timeframe trend supports mean reversion

name = "6h_1d_williamsr_macd_meanrev"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams %R on 6h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # MACD on 1d (12,26,9)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # need enough for slow EMA
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_fast = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Align 1d MACD to 6s
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)
    
    # Williams %R levels for entry
    wr_oversold = -80  # buy when below -80
    wr_overbought = -20  # sell when above -20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # start after Williams %R warmup
        # Skip if not ready
        if np.isnan(williams_r[i]) or np.isnan(macd_hist_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion conditions
        wr_buy = williams_r[i] < wr_oversold  # oversold
        wr_sell = williams_r[i] > wr_overbought  # overbought
        
        # MACD histogram filter: fade against the short-term momentum
        # In ranging markets, MACD histogram near zero; we fade extremes when momentum is weakening
        macd_weakening = abs(macd_hist_aligned[i]) < 0.1 * np.std(macd_hist_aligned[max(0, i-50):i+1]) if i >= 50 else True
        
        # Entry signals: fade extremes when momentum is weakening
        long_signal = wr_buy and macd_weakening
        short_signal = wr_sell and macd_weakening
        
        # Exit when Williams %R returns to neutral zone
        exit_long = williams_r[i] > -50
        exit_short = williams_r[i] < -50
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals