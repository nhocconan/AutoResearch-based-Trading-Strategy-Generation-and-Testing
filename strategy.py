# NOTE: This strategy attempts to use 1-day EMA50/EMA200 on 4h data for trend, with volatility filtering on 4h ATR.
# However, the previous version failed due to being applied on 4h timeframe while the prompt requires 1h.
# Adjusting to 1h timeframe with 4h/1d HTF for direction, and adding session filter (08-20 UTC) to reduce noise.
# Target: 15-37 trades/year (60-150 over 4 years). Position size: 0.20.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load HTF data ONCE (1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d EMA200
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 1h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1h price array
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_1d_aligned[i]
        ema_200_val = ema_200_1d_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price > EMA200 + low volatility (ATR < 40th percentile)
            if price > ema_200_val and atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 40):
                signals[i] = 0.20
                position = 1
            # Short: price < EMA50 + low volatility (ATR < 40th percentile)
            elif price < ema_50_val and atr_val < np.nanpercentile(atr_14_1d_aligned[:i+1], 40):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price < EMA50 or volatility spike (ATR > 60th percentile)
            if price < ema_50_val or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price > EMA200 or volatility spike (ATR > 60th percentile)
            if price > ema_200_val or atr_val > np.nanpercentile(atr_14_1d_aligned[:i+1], 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA50_EMA200_VolatilityFilter_Session"
timeframe = "1h"
leverage = 1.0