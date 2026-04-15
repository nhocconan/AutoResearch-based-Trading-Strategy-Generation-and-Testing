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
    
    # Get 4h HTF data once before loop (signal direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA21 for trend
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Get 1d HTF data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d ADX14 for trend strength
    plus_dm = np.diff(df_1d['high'], prepend=df_1d['high'][0])
    minus_dm = np.diff(df_1d['low'], prepend=df_1d['low'][0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr1 = np.abs(df_1d['high'] - df_1d['low'])
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_14_1d + 1e-10)
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_14_1d + 1e-10)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # 1h ATR14 for volatility filter
    tr1h = np.abs(high - low)
    tr2h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    trh = np.maximum(tr1h, np.maximum(tr2h, tr3h))
    atr_14_1h = pd.Series(trh).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1h volume ratio
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(adx_14_aligned[i]) or 
            np.isnan(atr_14_1h[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h EMA21 uptrend (price > EMA21)
        # 2. 1d ADX > 25 (strong trend)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.3% of price
        if (close[i] > ema_21_4h_aligned[i] and
            adx_14_aligned[i] > 25 and
            volume_ratio[i] > 1.5 and
            atr_14_1h[i] > 0.003 * close[i]):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 4h EMA21 downtrend (price < EMA21)
        # 2. 1d ADX > 25 (strong trend)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.3% of price
        elif (close[i] < ema_21_4h_aligned[i] and
              adx_14_aligned[i] > 25 and
              volume_ratio[i] > 1.5 and
              atr_14_1h[i] > 0.003 * close[i]):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_EMA21_1d_ADX25_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0