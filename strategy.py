#!/usr/bin/env python3
"""
1h_RSI_Trend_Pullback
1h strategy using 4h trend filter with RSI pullback entries.
- Long: 4h EMA34 up + RSI(14) pullback to 40-45 in uptrend
- Short: 4h EMA34 down + RSI(14) pullback to 55-60 in downtrend
- Exit: Opposite RSI extreme or trend change
Designed for ~15-30 trades/year per symbol (60-120 total over 4 years)
Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    close_4h = df_4h['close'].values
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough for EMA34
    
    for i in range(start_idx, n):
        # Skip if 4h EMA not available
        if np.isnan(ema_34_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1]  # rising EMA
        downtrend = ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1]  # falling EMA
        
        # RSI conditions
        rsi_now = rsi_values[i]
        rsi_pullback_long = 40 <= rsi_now <= 45
        rsi_pullback_short = 55 <= rsi_now <= 60
        rsi_overbought = rsi_now >= 70
        rsi_oversold = rsi_now <= 30
        
        if position == 0:
            # Long: uptrend + RSI pullback to 40-45
            if uptrend and rsi_pullback_long:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + RSI pullback to 55-60
            elif downtrend and rsi_pullback_short:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or trend change to down
            if rsi_overbought or downtrend:
                signals[i] = 0.0  # exit to flat
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI oversold or trend change to up
            if rsi_oversold or uptrend:
                signals[i] = 0.0  # exit to flat
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_Trend_Pullback"
timeframe = "1h"
leverage = 1.0