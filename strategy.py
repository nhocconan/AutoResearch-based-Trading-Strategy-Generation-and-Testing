#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation, 
trading only during 08-20 UTC session to reduce noise. Uses discrete position sizing (0.20) 
to minimize fee churn. Target: 15-30 trades/year per symbol to avoid fee drag while capturing 
strong intraday moves aligned with higher timeframe trend.
"""

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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla levels (using previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous 1d bar for Camarilla calculation
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Camarilla R1 and S1 levels from previous 1d bar
    R1 = close_1d_prev + (high_1d_prev - low_1d_prev) * 1.1 / 12
    S1 = close_1d_prev - (high_1d_prev - low_1d_prev) * 1.1 / 12
    
    # Align Camarilla levels (1d -> 1h)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 1.5x average volume (balanced for 1h timeframe)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24*1h = 24h ~ 1d
    
    # ATR for stoploss (using 14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    open_time = prices['open_time']
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 4h EMA (50), volume MA (24), ATR (14), 1d shift (1)
    start_idx = max(50, 24, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        session_ok = in_session[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation, uptrend, and in session
            long_signal = (high_val > R1_val) and (volume_val > 1.5 * vol_ma_val) and (close_val > ema_50_4h_val) and session_ok
            # Short: price breaks below S1 with volume confirmation, downtrend, and in session
            short_signal = (low_val < S1_val) and (volume_val > 1.5 * vol_ma_val) and (close_val < ema_50_4h_val) and session_ok
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: ATR stoploss or trend reversal or session end
            if (close_val < entry_price - 2.0 * atr_val or 
                close_val < ema_50_4h_val or 
                not session_ok):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: ATR stoploss or trend reversal or session end
            if (close_val > entry_price + 2.0 * atr_val or 
                close_val > ema_50_4h_val or 
                not session_ok):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0