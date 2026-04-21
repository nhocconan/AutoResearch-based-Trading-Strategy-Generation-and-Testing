#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily RSI(14) for mean reversion signal ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 4h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # === Daily ATR(10) for volatility regime ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # ATR(10) percentile (100-day lookback for regime)
    atr_percentile = pd.Series(atr_10).rolling(window=100, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # === Daily Close vs SMA200 for trend filter ===
    sma_200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(rsi_14_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(sma_200_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        rsi_val = rsi_14_aligned[i]
        atr_percentile_val = atr_percentile_aligned[i]
        sma_trend = sma_200_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long when oversold (RSI < 30) in low volatility + above SMA200 + volume
            if (rsi_val < 30 and 
                atr_percentile_val < 40 and  # Low volatility regime
                price_close > sma_trend and
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short when overbought (RSI > 70) in low volatility + below SMA200 + volume
            elif (rsi_val > 70 and 
                  atr_percentile_val < 40 and   # Low volatility regime
                  price_close < sma_trend and
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: RSI mean reversion or volatility expansion
            if position == 1 and (rsi_val > 50 or atr_percentile_val > 60):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (rsi_val < 50 or atr_percentile_val > 60):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_RSI_MeanReversion_VolatilityRegime_SMA200_Trend_Volume"
timeframe = "4h"
leverage = 1.0