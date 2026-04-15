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
    
    # 1d EMA(50) for trend filter
    daily = get_htf_data(prices, '1d')
    close_d = daily['close'].values
    ema_50d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50d_aligned = align_htf_to_ltf(prices, daily, ema_50d)
    
    # 1d ATR(20) for volatility filter
    high_d = daily['high'].values
    low_d = daily['low'].values
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - close_d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_20d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_20d_aligned = align_htf_to_ltf(prices, daily, atr_20d)
    
    # 1d volatility percentile (lookback 60 days) to filter extremes
    atr_series = pd.Series(atr_20d_aligned)
    atr_rank = atr_series.rolling(window=60, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Volume threshold: 2.0x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50d_aligned[i]) or np.isnan(atr_20d_aligned[i]) or
            np.isnan(atr_rank[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Volatility filter: avoid extreme volatility (bottom 20% and top 20%)
        vol_filter = (atr_rank[i] > 0.2) and (atr_rank[i] < 0.8)
        
        # Long: Price above daily EMA50 + volume spike + moderate volatility
        if (close[i] > ema_50d_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Price below daily EMA50 + volume spike + moderate volatility
        elif (close[i] < ema_50d_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price crosses back below/above daily EMA50
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < ema_50d_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > ema_50d_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DailyEMA50_Vol2.0x_VolRankFilter"
timeframe = "4h"
leverage = 1.0