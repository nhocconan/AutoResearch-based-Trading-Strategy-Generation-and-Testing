#!/usr/bin/env python3
"""
4h_1d_cci_trend_volume_v1
Strategy: 4h CCI trend with volume confirmation and 1d trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses CCI(20) on 4h to identify trend extremes (>100 for uptrend, <-100 for downtrend) with volume confirmation (>1.5x average volume) and filtered by 1d EMA50 trend alignment. Designed to capture strong trend moves while avoiding false signals in chop. Uses higher timeframe for direction (1d) and 4h only for signal generation. Target: 20-50 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_trend_volume_v1"
timeframe = "4h"
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
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h CCI(20)
    typical_price = (high + low + close) / 3
    ma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = (typical_price - ma_tp) / (0.015 * mad)
    cci = cci.values
    
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
        if (np.isnan(cci[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # CCI conditions
        cci_overbought = cci[i] > 100
        cci_oversold = cci[i] < -100
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: CCI > 100 with volume in uptrend
        long_signal = cci_overbought and vol_confirmed and uptrend_1d
        
        # Short: CCI < -100 with volume in downtrend
        short_signal = cci_oversold and vol_confirmed and downtrend_1d
        
        # Exit when CCI returns to neutral zone
        exit_long = position == 1 and cci[i] < 0
        exit_short = position == -1 and cci[i] > 0
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals