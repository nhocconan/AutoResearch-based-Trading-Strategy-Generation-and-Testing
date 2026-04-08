#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_obv_d1_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for OBV and trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate OBV (On-Balance Volume) on daily timeframe
    price_change_1d = np.diff(close_1d, prepend=close_1d[0])
    obv_1d = np.cumsum(np.where(price_change_1d > 0, volume_1d, np.where(price_change_1d < 0, -volume_1d, 0)))
    
    # EMA21 of OBV for trend filter
    obv_ema21 = pd.Series(obv_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    obv_ema21_aligned = align_htf_to_ltf(prices, df_1d, obv_ema21)
    
    # Raw OBV for divergence check
    obv_aligned = align_htf_to_ltf(prices, df_1d, obv_1d)
    
    # 60-period EMA on 6h for price trend filter
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # ATR for volatility filter (24-period on 6h)
    tr1 = pd.Series(high).subtract(pd.Series(low)).abs()
    tr2 = pd.Series(high).subtract(pd.Series(close).shift(1)).abs()
    tr3 = pd.Series(low).subtract(pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=24, min_periods=24).mean().values
    atr_ma = pd.Series(atr).rolling(window=30, min_periods=30).mean().values
    vol_filter = atr > atr_ma  # Avoid low volatility chop
    
    # Volume filter: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(obv_ema21_aligned[i]) or 
            np.isnan(obv_aligned[i]) or
            np.isnan(ema_60[i]) or np.isnan(vol_spike[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: OBV turns down OR price breaks below EMA60
            if obv_aligned[i] < obv_ema21_aligned[i] or close[i] < ema_60[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: OBV turns up OR price breaks above EMA60
            if obv_aligned[i] > obv_ema21_aligned[i] or close[i] > ema_60[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: OBV rising above EMA21 + price above EMA60 + volume spike + vol filter
            if (obv_aligned[i] > obv_ema21_aligned[i] and 
                close[i] > ema_60[i] and
                vol_spike[i] and
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: OBV falling below EMA21 + price below EMA60 + volume spike + vol filter
            elif (obv_aligned[i] < obv_ema21_aligned[i] and 
                  close[i] < ema_60[i] and
                  vol_spike[i] and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals