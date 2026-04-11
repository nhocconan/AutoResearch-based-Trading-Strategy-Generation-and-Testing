#!/usr/bin/env python3
"""
1h_4h_1d_cci_momentum_v1
Strategy: 1h CCI momentum with volume confirmation and 4h/1d trend filter
Timeframe: 1h
Leverage: 1.0
Hypothesis: Uses CCI (Commodity Channel Index) to identify momentum extremes, filtered by 4h and 1d EMA trend alignment and volume spikes. CCI > 100 indicates strong uptrend momentum, CCI < -100 indicates strong downtrend momentum. Uses higher timeframes for direction (4h/1d) and 1h only for timing. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_cci_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # CCI(20) calculation
    typical_price = (high + low + close) / 3
    tp_ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - tp_ma) / (0.015 * tp_mad)
    cci_values = cci.values
    
    # 20-period EMA for 1h trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cci_values[i]) or np.isnan(ema_20[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        price_close = close[i]
        cci_val = cci_values[i]
        
        # Trend filters: price above/both EMAs for long, below/both for short
        uptrend_4h = price_close > ema_20_4h_aligned[i]
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_4h = price_close < ema_20_4h_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # CCI momentum conditions
        strong_uptrend = cci_val > 100
        strong_downtrend = cci_val < -100
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: strong uptrend momentum with volume in uptrend (both 4h and 1d)
        long_signal = strong_uptrend and vol_confirmed and uptrend_4h and uptrend_1d
        
        # Short: strong downtrend momentum with volume in downtrend (both 4h and 1d)
        short_signal = strong_downtrend and vol_confirmed and downtrend_4h and downtrend_1d
        
        # Exit when CCI returns to neutral zone or trend changes
        exit_long = position == 1 and (cci_val < 0 or not (uptrend_4h and uptrend_1d))
        exit_short = position == -1 and (cci_val > 0 or not (downtrend_4h and downtrend_1d))
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals