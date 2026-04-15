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
    
    # Calculate 1d RSI(14)
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 1d ADX(14)
    plus_dm = pd.Series(df_1d['high'].values).diff()
    minus_dm = pd.Series(df_1d['low'].values).diff().copy()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    plus_di_14 = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / 
                        pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean())
    minus_di_14 = 100 * (-minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / 
                         pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean())
    dx = (np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)) * 100
    adx_14_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate 6h Donchian(20) channels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_1d_aligned[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade when trend is strong (ADX > 25)
        trend_filter = adx_14_1d_aligned[i] > 25
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_not_extreme = (rsi_14_1d_aligned[i] > 30) and (rsi_14_1d_aligned[i] < 70)
        
        # Long conditions:
        # 1. Price breaks above 6h Donchian(20) high
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Daily trend filter (ADX > 25)
        # 4. Daily RSI not extreme (avoid buying into overbought)
        if (close[i] > donchian_high_20[i] and
            volume_ratio[i] > 1.5 and
            trend_filter and
            rsi_not_extreme):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 6h Donchian(20) low
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Daily trend filter (ADX > 25)
        # 4. Daily RSI not extreme (avoid selling into oversold)
        elif (close[i] < donchian_low_20[i] and
              volume_ratio[i] > 1.5 and
              trend_filter and
              rsi_not_extreme):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ADX25_Donchian20_Volume_Breakout_RSIFilter_v1"
timeframe = "6h"
leverage = 1.0