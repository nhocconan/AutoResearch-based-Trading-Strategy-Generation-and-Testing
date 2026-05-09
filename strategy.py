#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_MultiFactor_Breakout_v1"
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
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === PRE-COMPUTE INDICATORS OUTSIDE LOOP ===
    
    # 1d EMA for trend filter
    ema20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 1d ATR for volatility filter
    tr1 = np.maximum(df_1d['high'], df_1d['close'].shift(1)) - np.minimum(df_1d['low'], df_1d['close'].shift(1))
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 4h volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # 4h price change momentum (3-period ROC)
    roc_series = pd.Series(close)
    roc = roc_series.pct_change(periods=3) * 100
    roc_values = roc.values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 50)  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(atr10_1d_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(roc_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        dh = donch_high[i]
        dl = donch_low[i]
        trend = ema20_1d_aligned[i]
        atr = atr10_1d_aligned[i]
        vol_ok = volume_filter[i]
        roc_val = roc_values[i]
        
        # Dynamic thresholds based on volatility
        atr_mult = 0.5  # Multiple of ATR for breakout threshold
        
        if position == 0:
            # Long entry: price breaks above Donchian high with volume, trend alignment, and positive momentum
            if (close[i] > dh + atr * atr_mult and 
                close[i] > trend and 
                vol_ok and 
                roc_val > 0):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume, trend alignment, and negative momentum
            elif (close[i] < dl - atr * atr_mult and 
                  close[i] < trend and 
                  vol_ok and 
                  roc_val < 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend deteriorates
            if close[i] < dl or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend deteriorates
            if close[i] > dh or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals