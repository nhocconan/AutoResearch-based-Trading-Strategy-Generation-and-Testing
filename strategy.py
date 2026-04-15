#!/usr/bin/env python3
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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d ADX(14) for trend strength filter
    plus_dm = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']), 
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    minus_dm = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)), 
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx_14).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    camarilla_pivot = (prior_high + prior_low + prior_close) / 3.0
    camarilla_r3 = camarilla_pivot + 1.1 * (prior_high - prior_low)
    camarilla_s3 = camarilla_pivot - 1.1 * (prior_high - prior_low)
    
    # Align Camarilla levels to 6h
    camarilla_pivot_6h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(adx_14_aligned[i]) or 
            np.isnan(camarilla_pivot_6h[i]) or np.isnan(camarilla_r3_6h[i]) or 
            np.isnan(camarilla_s3_6h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility and trend regime filters: only trade when daily ATR is elevated AND ADX > 25
        # This avoids low-volatility chop and range-bound markets
        vol_regime = atr_14_1d_aligned[i] > 0.006 * close[i]
        trend_regime = adx_14_aligned[i] > 25
        
        # Long conditions:
        # 1. Price above Camarilla pivot (bullish bias)
        # 2. Price breaks above Camarilla R3 with volume (bullish continuation)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility regime filter
        # 5. Trend regime filter (ADX > 25)
        if (close[i] > camarilla_pivot_6h[i] and
            close[i] > camarilla_r3_6h[i] and
            volume_ratio[i] > 1.5 and
            vol_regime and
            trend_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below Camarilla pivot (bearish bias)
        # 2. Price breaks below Camarilla S3 with volume (bearish continuation)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility regime filter
        # 5. Trend regime filter (ADX > 25)
        elif (close[i] < camarilla_pivot_6h[i] and
              close[i] < camarilla_s3_6h[i] and
              volume_ratio[i] > 1.5 and
              vol_regime and
              trend_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Vol_Trend_Regime_Camarilla_Pivot_R3S3_Breakout_v1"
timeframe = "6h"
leverage = 1.0