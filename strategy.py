#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # === Daily RSI(14) ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === Daily ATR(14) for volatility regime ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_14 = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # === Daily EMA(34) for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        rsi_val = rsi_1d_aligned[i]
        atr_val = atr_14_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long in low volatility + oversold RSI + uptrend + volume
            if (atr_val < np.percentile(atr_14_aligned[:i+1], 30) and  # Low volatility regime
                rsi_val < 30 and  # Oversold
                price_close > ema_trend and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short in low volatility + overbought RSI + downtrend + volume
            elif (atr_val < np.percentile(atr_14_aligned[:i+1], 30) and  # Low volatility regime
                  rsi_val > 70 and  # Overbought
                  price_close < ema_trend and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when volatility increases (trending regime) or RSI reverts
            if position == 1 and (atr_val > np.percentile(atr_14_aligned[:i+1], 70) or rsi_val > 70):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (atr_val > np.percentile(atr_14_aligned[:i+1], 70) or rsi_val < 30):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_RSI_MeanReversion_ATR_Volatility_Regime"
timeframe = "12h"
leverage = 1.0