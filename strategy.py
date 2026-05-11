#!/usr/bin/env python3
"""
1H_Trend_Follow_4H1D_Confirm
Hypothesis: 1h trend following with 4h/1d confirmation to avoid false breakouts.
Uses 4h EMA21 for trend direction and 1d ATR for volatility filter.
Trades only during London/NY session (08-20 UTC) to reduce noise.
Designed for 15-35 trades/year with 0.20 position size to minimize fee drag.
Works in bull/bear via trend filter and volatility-adjusted entries.
"""

name = "1H_Trend_Follow_4H1D_Confirm"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Precompute session filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h EMA21 for trend filter ---
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # --- 1d ATR14 for volatility filter ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d, additional_delay_bars=0)
    
    # --- 1h EMA21 for entry timing ---
    ema_21_1h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(ema_21_1h[i]) or
            not in_session[i]):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: 
        # Long: price > 4h EMA21 AND price > 1h EMA21 AND volatility normal
        # Short: price < 4h EMA21 AND price < 1h EMA21 AND volatility normal
        vol_normal = atr_14_1d_aligned[i] < np.percentile(atr_14_1d_aligned[:i+1], 80)
        
        long_entry = (close[i] > ema_21_4h_aligned[i]) and (close[i] > ema_21_1h[i]) and vol_normal
        short_entry = (close[i] < ema_21_4h_aligned[i]) and (close[i] < ema_21_1h[i]) and vol_normal
        
        if position == 0:
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: 
            # Long: price < 1h EMA21 OR volatility spike
            # Short: price > 1h EMA21 OR volatility spike
            vol_spike = atr_14_1d_aligned[i] > np.percentile(atr_14_1d_aligned[:i+1], 90)
            
            if position == 1:
                if (close[i] < ema_21_1h[i]) or vol_spike:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                if (close[i] > ema_21_1h[i]) or vol_spike:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals