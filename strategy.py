#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_ChopRegime_v2"
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
    
    # Get 1d data for Chop and TRIX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # TRIX on 1d close: triple EMA
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    
    # Chop on 1d: true range and ATR
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Chop calculation
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(14)
    
    # Align TRIX and Chop to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_raw)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: current 4h volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 14, 12)  # Need enough data
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix = trix_aligned[i]
        chop_val = chop_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Chop > 61.8 = ranging (mean revert)
            if chop_val > 61.8 and vol_filter:
                if trix < -0.5:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif trix > 0.5:  # Overbought
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: TRIX crosses above zero or Chop < 38.2 (trending)
            if trix > 0.0 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses below zero or Chop < 38.2
            if trix < 0.0 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#%%