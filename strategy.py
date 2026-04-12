#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX and volume context (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # TRIX calculation on daily close (15-period EMA triple)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix_raw.values
    
    # Align TRIX to 4h timeframe (with extra delay for momentum confirmation)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix, additional_delay_bars=1)
    
    # Daily volume average for context
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=10, min_periods=10).mean().values
    vol_ratio_1d = vol_1d / vol_ma_1d
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 4h price momentum (5-period ROC)
    close_series = pd.Series(close)
    roc_5 = close_series.pct_change(5) * 100
    roc_5_vals = roc_5.values
    
    # 4h volume filter (20-period average)
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ok = volume > vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_ratio_aligned[i]) or
            np.isnan(roc_5_vals[i]) or np.isnan(vol_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: TRIX momentum + volume confirmation + price alignment
        long_setup = (trix_aligned[i] > 0.05 and  # Positive TRIX momentum
                     vol_ratio_aligned[i] > 1.2 and  # Above average daily volume
                     roc_5_vals[i] > 0 and  # Positive 4h momentum
                     vol_ok[i])  # 4h volume confirmation
        
        short_setup = (trix_aligned[i] < -0.05 and  # Negative TRIX momentum
                      vol_ratio_aligned[i] > 1.2 and  # Above average daily volume
                      roc_5_vals[i] < 0 and  # Negative 4h momentum
                      vol_ok[i])  # 4h volume confirmation
        
        # Exit when TRIX momentum fades or reverses
        exit_long = trix_aligned[i] < 0  # TRIX turns negative
        exit_short = trix_aligned[i] > 0  # TRIX turns positive
        
        # Execute trades
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals