#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_CCI_Momentum_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period CCI on weekly
    # Typical Price
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # 20-period SMA of TP
    tp_sma_20 = pd.Series(tp_1w).rolling(window=20, min_periods=20).mean().values
    
    # Mean Deviation
    tp_dev = np.abs(tp_1w - tp_sma_20)
    tp_md_20 = pd.Series(tp_dev).rolling(window=20, min_periods=20).mean().values
    
    # CCI
    cci_1w = (tp_1w - tp_sma_20) / (0.015 * tp_md_20)
    
    # Align CCI to daily
    cci_1w_aligned = align_htf_to_ltf(prices, df_1w, cci_1w)
    
    # Daily 50-period EMA for trend filter
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        if np.isnan(cci_1w_aligned[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        cci = cci_1w_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend filter
        above_ema = price > ema_50[i]
        below_ema = price < ema_50[i]
        
        if position == 0:
            # Long: CCI > 100 (strong momentum) + above EMA + volume
            if cci > 100 and above_ema and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: CCI < -100 (strong bearish momentum) + below EMA + volume
            elif cci < -100 and below_ema and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: CCI drops below 0 or price crosses below EMA
            if cci < 0 or not above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: CCI rises above 0 or price crosses above EMA
            if cci > 0 or not below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals