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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d Donchian(20) channels
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 1d RSI(14) for momentum filter
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or np.isnan(rsi_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when daily ATR is elevated (> 0.3% of price)
        vol_filter = atr_14_1d_aligned[i] > 0.003 * close[i]
        
        # Long conditions:
        # 1. Price breaks above daily Donchian(20) high
        # 2. Daily RSI > 50 (bullish momentum)
        # 3. Volume confirmation: volume > 1.5x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        volume_confirm = volume[i] > 1.5 * vol_ma_20
        
        if (close[i] > donchian_high_20_aligned[i] and
            rsi_14_1d_aligned[i] > 50 and
            volume_confirm and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below daily Donchian(20) low
        # 2. Daily RSI < 50 (bearish momentum)
        # 3. Volume confirmation: volume > 1.5x 20-period average
        elif (close[i] < donchian_low_20_aligned[i] and
              rsi_14_1d_aligned[i] < 50 and
              volume_confirm and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_RSI14_Volume_Breakout_v1"
timeframe = "4h"
leverage = 1.0