#!/usr/bin/env python3
"""
1d_KeltnerChannel_Breakout_With_TrendFilter
Hypothesis: Keltner Channel breakouts on daily timeframe with weekly trend filter.
Buy when price closes above upper KC with weekly EMA uptrend, short when closes below lower KC with weekly EMA downtrend.
Designed for low trade frequency (<25/year) to minimize fee decay while capturing sustained trends in both bull and bear markets.
Uses volatility-based channel (ATR) which adapts to market conditions, reducing false breakouts in ranging periods.
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(34) for trend filter
    close_weekly = df_weekly['close'].values
    ema_34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_34_weekly)
    
    # Daily ATR(14) for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA(20) for KC middle line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel: EMA(20) ± 2 * ATR(14)
    kc_upper = ema_20 + 2.0 * atr_14
    kc_lower = ema_20 - 2.0 * atr_14
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_weekly_aligned[i]) or 
            np.isnan(atr_14[i]) or
            np.isnan(kc_upper[i]) or
            np.isnan(kc_lower[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_ema = ema_34_weekly_aligned[i]
        upper = kc_upper[i]
        lower = kc_lower[i]
        
        if position == 0:
            # Long: close above upper KC with weekly uptrend
            if price > upper and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: close below lower KC with weekly downtrend
            elif price < lower and price < weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: close below middle line OR weekly trend turns down
            if price < ema_20[i] or price < weekly_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: close above middle line OR weekly trend turns up
            if price > ema_20[i] or price > weekly_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KeltnerChannel_Breakout_With_TrendFilter"
timeframe = "1d"
leverage = 1.0