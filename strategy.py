# Your solution is ready. Here is the complete, valid python code.  
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_ChopFilter"
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
    
    # Get 1d data for TRIX and Chop (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate TRIX on 1d close
    def ema(arr, span):
        return pd.Series(arr).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema1 = ema(close_1d, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    trix_raw = np.where(ema3 != 0, (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100, 0)
    trix_raw[0] = 0
    trix = ema(trix_raw, 9)
    
    # Chop: True Range and Chop calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_1d.sum() / (max_hh - min_ll)) / np.log10(14)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    
    # Align TRIX and Chop to 4h timeframe
    trix_4h = align_htf_to_ltf(prices, df_1d, trix)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h ATR for volatility and stop loss
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=15, min_periods=15).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(trix_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_4h[i]
        trix_val = trix_4h[i]
        chop_val = chop_4h[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        chop_filter = chop_val > 50  # Chop > 50 indicates ranging/transitional market
        
        if position == 0:
            # Long: TRIX crosses above 0 + volume + chop filter
            if trix_val > 0 and np.roll(trix_4h, 1)[i] <= 0 and volume_confirmed and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below 0 + volume + chop filter
            elif trix_val < 0 and np.roll(trix_4h, 1)[i] >= 0 and volume_confirmed and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below 0 OR ATR stop (2.0x ATR from entry)
            if trix_val < 0 or price < (high[i] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above 0 OR ATR stop (2.0x ATR from entry)
            if trix_val > 0 or price > (low[i] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals