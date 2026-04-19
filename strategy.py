#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Keltner_MeanReversion_4hTrend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(20) for trend
    close_4h_series = pd.Series(close_4h)
    ema20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate 4h ATR(14) for Keltner width
    tr_4h = np.maximum(np.abs(high[1:] - low[1:]), np.abs(high[1:] - close_4h[:-1]))
    tr_4h = np.maximum(tr_4h, np.abs(low[1:] - close_4h[:-1]))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr14_4h)
    
    # Calculate 1h EMA(20) for Keltner center
    close_series = pd.Series(close)
    ema20_1h = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner Bands (20, 2.0)
    upper_keltner = ema20_1h + 2.0 * atr14_4h_aligned
    lower_keltner = ema20_1h - 2.0 * atr14_4h_aligned
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Time filter: 08-20 UTC (pre-market to post-US session)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    time_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(atr14_4h_aligned[i]) or 
            np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(time_filter[i])):
            signals[i] = 0.0
            continue
        
        # Check time filter
        if not time_filter[i]:
            signals[i] = 0.0
            continue
            
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_4h = ema20_4h_aligned[i]
        upper = upper_keltner[i]
        lower = lower_keltner[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price below lower Keltner + 4h uptrend + volume
            if price < lower and price > ema_4h and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: price above upper Keltner + 4h downtrend + volume
            elif price > upper and price < ema_4h and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price back to EMA(20) or opposite band touch
            if price >= ema20_1h[i] or price >= upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price back to EMA(20) or opposite band touch
            if price <= ema20_1h[i] or price <= lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals