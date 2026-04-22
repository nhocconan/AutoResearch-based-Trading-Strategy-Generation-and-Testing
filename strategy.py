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
    
    # Load 1d data (primary timeframe) and 1w data (HTF) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly ATR (14-period) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr1_w[0] = 0
    tr2_w[0] = 0
    tr3_w[0] = 0
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    atr_14_1w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to daily timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1w)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # 1d RSI (14-period) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = rsi  # already on 1d timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_14_1w_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: only trade when 1d ATR is above weekly ATR (high volatility regime)
        vol_regime = atr_14_aligned[i] > atr_14_1w_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) + volume spike + volatility regime
            if (rsi_aligned[i] < 30 and 
                volume[i] > 2.0 * vol_avg_20_aligned[i] and 
                vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + volume spike + volatility regime
            elif (rsi_aligned[i] > 70 and 
                  volume[i] > 2.0 * vol_avg_20_aligned[i] and 
                  vol_regime):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or volatility regime ends
            if position == 1:
                if rsi_aligned[i] > 40 or not vol_regime:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi_aligned[i] < 60 or not vol_regime:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_RSI14_VolumeSpike_VolatilityRegime"
timeframe = "1d"
leverage = 1.0