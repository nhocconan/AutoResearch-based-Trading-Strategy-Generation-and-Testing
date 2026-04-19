#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d1d_TurtleBreakout_3atm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h 20-period Donchian channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    
    # 1d 50-period EMA for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h ATR(14) for volatility filter
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr2])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema = ema_50_aligned[i]
        volatility = atr[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_ok = volatility > 0.001 * price  # at least 0.1% of price
        
        if position == 0 and in_session and vol_ok:
            # Long: price breaks above 20-period 4h high AND above 1d EMA50
            if price > upper and price > ema:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 20-period 4h low AND below 1d EMA50
            elif price < lower and price < ema:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price breaks below 20-period 4h low OR below 1d EMA50
            if price < lower or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price breaks above 20-period 4h high OR above 1d EMA50
            if price > upper or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals