#%%
#!/usr/bin/env python3
"""
1d_1w_Volatility_Breakout_with_Trend_Filter
Hypothesis: Breakouts of weekly volatility-adjusted channels on daily timeframe, filtered by weekly trend.
Uses weekly ATR-based channel (donchian-like) and weekly EMA trend filter to avoid false breakouts.
Designed for 1d timeframe to capture multi-day moves with low trade frequency.
Works in both bull and bear markets by aligning with weekly trend direction.
Target: 10-25 trades/year on 1d.
"""

name = "1d_1w_Volatility_Breakout_with_Trend_Filter"
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
    
    # === Weekly ATR-based Channel ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly ATR (14-period)
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    # True Range
    tr1 = wk_high[1:] - wk_low[1:]
    tr2 = np.abs(wk_high[1:] - wk_close[:-1])
    tr3 = np.abs(wk_low[1:] - wk_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14)
    atr_14 = np.full_like(wk_close, np.nan)
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_14[i] = np.nanmean(tr[i-13:i+1])
    
    # Weekly channel: ±1.5 * ATR from weekly close
    channel_width = 1.5 * atr_14
    wk_upper = wk_close + channel_width
    wk_lower = wk_close - channel_width
    
    # Align weekly channel to daily
    upper_daily = align_htf_to_ltf(prices, df_1w, wk_upper)
    lower_daily = align_htf_to_ltf(prices, df_1w, wk_lower)
    
    # === Weekly Trend Filter (EMA20) ===
    ema20_1w = pd.Series(wk_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_daily = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_daily[i]) or np.isnan(lower_daily[i]) or 
            np.isnan(ema20_daily[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper channel with weekly uptrend
            if (close[i] > upper_daily[i] and 
                close[i] > ema20_daily[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below lower channel with weekly downtrend
            elif (close[i] < lower_daily[i] and 
                  close[i] < ema20_daily[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below lower channel (reversal)
            if close[i] < lower_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above upper channel (reversal)
            if close[i] > upper_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
#%%