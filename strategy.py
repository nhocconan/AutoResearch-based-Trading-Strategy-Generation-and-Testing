#!/usr/bin/env python3
"""
1d_volatility_breakout_1w_trend_v1
Hypothesis: On 1d timeframe, trade volatility breakouts aligned with weekly trend. 
Go long when price breaks above 20-day high with expanding volume in a low volatility regime (volatility < 50th percentile) and weekly trend is up (price > weekly EMA20). 
Go short when price breaks below 20-day low with expanding volume in low volatility regime and weekly trend is down (price < weekly EMA20). 
Use 1d ATR percentile to filter for low volatility environments where breakouts are more likely to succeed. 
Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag and improve generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_volatility_breakout_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-day high and low for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR for volatility regime filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR percentile rank (252-day lookback for ~1 year)
    atr_percentile = pd.Series(atr).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(atr_percentile[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime: ATR percentile below 50th percentile
        low_vol = atr_percentile[i] < 0.5
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous 20-day high
        breakout_down = close[i] < low_20[i-1]  # Break below previous 20-day low
        
        if position == 1:  # Long position
            # Exit: price crosses back below 20-day low or volatility increases significantly
            if close[i] < low_20[i] or atr_percentile[i] > 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above 20-day high or volatility increases significantly
            if close[i] > high_20[i] or atr_percentile[i] > 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if low_vol and vol_ok:
                # Breakout above 20-day high with volume - go long (only if weekly trend up)
                if breakout_up and close[i] > ema_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout below 20-day low with volume - go short (only if weekly trend down)
                elif breakout_down and close[i] < ema_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals