#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d TRIX (12-period EMA of EMA of EMA)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Triple EMA
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    
    # TRIX = (EMA3 - prev EMA3) / prev EMA3 * 100
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix = trix.fillna(0).values  # Handle initial NaN
    
    # Calculate 1d average volume (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate 4h ADX for trend strength (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period_adx = 14
    alpha = 1.0 / period_adx
    tr14 = np.zeros_like(tr)
    plus_dm14 = np.zeros_like(plus_dm)
    minus_dm14 = np.zeros_like(minus_dm)
    
    tr14[period_adx] = tr[1:period_adx+1].sum()
    plus_dm14[period_adx] = plus_dm[1:period_adx+1].sum()
    minus_dm14[period_adx] = minus_dm[1:period_adx+1].sum()
    
    for i in range(period_adx + 1, len(tr)):
        tr14[i] = tr14[i-1] - (tr14[i-1] / period_adx) + tr[i]
        plus_dm14[i] = plus_dm14[i-1] - (plus_dm14[i-1] / period_adx) + plus_dm[i]
        minus_dm14[i] = minus_dm14[i-1] - (minus_dm14[i-1] / period_adx) + minus_dm[i]
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    
    # Smooth DX to get ADX
    adx = np.zeros_like(dx)
    adx[2*period_adx] = dx[period_adx+1:2*period_adx+1].mean()
    for i in range(2*period_adx + 1, len(dx)):
        adx[i] = (adx[i-1] * (period_adx - 1) + dx[i]) / period_adx
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 40 to ensure sufficient data
    for i in range(40, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_surge = vol_1d_current > 1.3 * vol_avg_20_1d_aligned[i]  # 30% above average
        
        # Long when TRIX > 0 with volume surge and ADX > 20 (trending)
        long_signal = trix_aligned[i] > 0 and vol_surge and adx_aligned[i] > 20
        # Short when TRIX < 0 with volume surge and ADX > 20 (trending)
        short_signal = trix_aligned[i] < 0 and vol_surge and adx_aligned[i] > 20
        
        # Exit when TRIX crosses zero (mean reversion in ranging markets)
        exit_long = trix_aligned[i] < 0
        exit_short = trix_aligned[i] > 0
        
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
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals