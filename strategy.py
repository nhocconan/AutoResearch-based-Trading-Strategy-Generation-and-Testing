#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_1dTrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # TRIX (15-period EMA applied 3 times)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = ema3.pct_change() * 100
    trix_signal = trix_raw.ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume spike filter: current volume > 2 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(trix_signal[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        trix_val = trix_signal[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        close_val = close[i]
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike and price above daily EMA34
            if trix_val > 0 and trix_signal[i-1] <= 0 and vol_spike and close_val > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike and price below daily EMA34
            elif trix_val < 0 and trix_signal[i-1] >= 0 and vol_spike and close_val < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if trix_val < 0 and trix_signal[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if trix_val > 0 and trix_signal[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals