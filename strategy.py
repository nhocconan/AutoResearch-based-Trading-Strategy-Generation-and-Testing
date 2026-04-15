#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and ATR-based position sizing
# Works in bull (breakouts with trend) and bear (filters out false breakouts, avoids counter-trend)
# Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 20-40 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily ATR(14) for volatility filter and position sizing
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when daily ATR > 0.3% of price
        vol_filter = atr_14_1d_aligned[i] > 0.003 * close[i]
        
        # Long conditions:
        # 1. Price breaks above 4h Donchian upper channel
        # 2. Price above daily EMA50 (bullish bias)
        # 3. Volatility filter
        if (close[i] > highest_high[i] and 
            close[i] > ema_50_1d_aligned[i] and 
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 4h Donchian lower channel
        # 2. Price below daily EMA50 (bearish bias)
        # 3. Volatility filter
        elif (close[i] < lowest_low[i] and 
              close[i] < ema_50_1d_aligned[i] and 
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_EMA50_VolFilter_v1"
timeframe = "4h"
leverage = 1.0