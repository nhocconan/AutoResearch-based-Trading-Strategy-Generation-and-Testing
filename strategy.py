#!/usr/bin/env python3
name = "1d_Pullback_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # EMA for weekly trend
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily EMA for pullback
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Average True Range for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(ema_50[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend condition: weekly EMA20 slope
        trend_up = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]
        trend_down = ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]
        
        # Pullback condition: price near EMA50 with EMA20 above/below
        pullback_long = (close[i] > ema_50[i]) and (ema_20[i] > ema_50[i]) and (close[i] < ema_20[i])
        pullback_short = (close[i] < ema_50[i]) and (ema_20[i] < ema_50[i]) and (close[i] > ema_20[i])
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr[i] > np.nanmedian(atr[max(0, i-50):i+1])
        
        if position == 0:
            # Long: pullback in uptrend
            if trend_up and pullback_long and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: pullback in downtrend
            elif trend_down and pullback_short and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend reversal or stop
            if not trend_up or close[i] < ema_20[i] - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend reversal or stop
            if not trend_down or close[i] > ema_20[i] + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Pullback to EMA50 in the direction of weekly trend
# - In weekly uptrend (EMA20 rising): buy pullbacks when price touches EMA50 from below with EMA20 > EMA50
# - In weekly downtrend (EMA20 falling): sell pullbacks when price touches EMA50 from above with EMA20 < EMA50
# - Uses EMA20/50 for dynamic support/resistance and trend definition
# - Volatility filter ensures trades occur in sufficient volatility environments
# - Stop loss at 1.5x ATR from EMA20 to limit losses
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Works in both bull and bear markets by following weekly trend
# - Simple, robust logic with minimal overfitting risk