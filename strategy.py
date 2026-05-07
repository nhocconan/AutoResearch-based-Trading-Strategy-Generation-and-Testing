#!/usr/bin/env python3
name = "1h_4h1d_Trend_Follow_With_Volume_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA(100) for higher timeframe trend filter
    ema_100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # 1h volume spike detection: 24-period average (1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 1h ATR(14) for volatility filter
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.inf], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 100, 24, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(vol_ma_24[i]) or np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_14[i] > 0.01 * close[i]  # At least 1% of price
        
        if position == 0:
            # Long: 4h uptrend + 1d uptrend + volume spike
            trend_4h = ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]
            trend_1d = ema_100_1d_aligned[i] > ema_100_1d_aligned[i-1]
            vol_condition = volume[i] > vol_ma_24[i] * 1.8
            
            if trend_4h and trend_1d and vol_condition and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + 1d downtrend + volume spike
            elif not trend_4h and not trend_1d and vol_condition and vol_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: 4h trend breaks or volume drops significantly
            trend_4h = ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]
            vol_condition = volume[i] > vol_ma_24[i] * 1.2
            
            if not trend_4h or not vol_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: 4h trend breaks or volume drops significantly
            trend_4h = ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]
            vol_condition = volume[i] > vol_ma_24[i] * 1.2
            
            if trend_4h or not vol_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h trend following with 4h/1d trend alignment and volume confirmation
# - Uses 4h EMA(50) for intermediate trend direction
# - Uses 1d EMA(100) for higher timeframe trend filter (avoid counter-trend trades)
# - Requires volume spike (1.8x average) for entry confirmation
# - Session filter (08-20 UTC) to focus on active trading hours
# - Volatility filter to avoid choppy markets
# - Position size 0.20 to manage risk and reduce trade frequency
# - Designed to work in both bull (follow uptrends) and bear (follow downtrends) markets
# - Target: 15-30 trades/year per symbol to avoid fee drag while capturing trends