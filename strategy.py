#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values  # datetime64[ms]
    
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
    
    # Calculate daily KAMA(10,2,30) - Kaufman Adaptive Moving Average
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(df_1d['close'].diff(10).values)
    volatility = np.sum(np.abs(df_1d['close'].diff(1).values), axis=0)  # will fix below
    # Recalculate volatility properly
    volatility = np.zeros(len(df_1d))
    for i in range(10, len(df_1d)):
        volatility[i] = np.sum(np.abs(df_1d['close'].iloc[i-9:i+1].diff(1).values))
    
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(len(df_1d))
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 1d RSI(14)
    delta = df_1d['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14.values)
    
    # Calculate 1d ADX(14) for trend strength
    plus_dm = df_1d['high'].diff()
    minus_dm = df_1d['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    plus_di_14 = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr_14)
    minus_di_14 = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr_14)
    dx = (np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)) * 100
    adx_14 = dx.rolling(window=14, min_periods=14).mean()
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14.values)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_14_aligned[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: only trade when daily ATR is elevated (> 0.5% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.005 * close[i]
        
        # Trend regime: only trade when ADX > 20 (trending market)
        trend_regime = adx_14_aligned[i] > 20
        
        # Long conditions:
        # 1. Price above KAMA (bullish bias)
        # 2. RSI > 50 (bullish momentum)
        # 3. Volatility regime filter
        # 4. Trend regime filter
        if (close[i] > kama_aligned[i] and
            rsi_14_aligned[i] > 50 and
            vol_regime and
            trend_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below KAMA (bearish bias)
        # 2. RSI < 50 (bearish momentum)
        # 3. Volatility regime filter
        # 4. Trend regime filter
        elif (close[i] < kama_aligned[i] and
              rsi_14_aligned[i] < 50 and
              vol_regime and
              trend_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_ADX_VolRegime_TrendFollow_v1"
timeframe = "1d"
leverage = 1.0