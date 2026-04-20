#!/usr/bin/env python3
# Strategy: 1h_4h_1d_Camarilla_R1S1_Breakout_Volume_RSIFilter_v1
# Hypothesis: Use 1d Camarilla R1/S1 for directional bias, 4h volume and RSI for regime filter, 1h for precise entry timing.
# Only trade when 1h price breaks daily R1/S1 with volume > 2x 20-period MA and 4h RSI not extreme.
# Target: 15-37 trades/year by combining multi-timeframe filters to reduce noise.
# Works in bull/bear: RSI filter avoids chasing extremes, volume confirms institutional interest.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for directional bias (Camarilla levels, RSI)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = np.where(np.isfinite(rsi_14), rsi_14, 100.0)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Daily Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R1 = pivot_1d + (range_1d * 1.1 / 12)
    S1 = pivot_1d - (range_1d * 1.1 / 12)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Load 4h data for regime filter (volume, RSI)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h RSI(14)
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_4h = avg_gain_4h / avg_loss_4h
    rsi_14_4h = 100 - (100 / (1 + rs_4h))
    rsi_14_4h = np.where(np.isfinite(rsi_14_4h), rsi_14_4h, 100.0)
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # 4h volume MA(20)
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(rsi_14_4h_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]  # 1h close
        
        if position == 0:
            # Long: price breaks above R1, 4h RSI not overbought (<60), volume confirmation
            if (price > R1_aligned[i] and 
                rsi_14_4h_aligned[i] < 60 and 
                prices['volume'].iloc[i] > 2.0 * vol_ma_20_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1, 4h RSI not oversold (>40), volume confirmation
            elif (price < S1_aligned[i] and 
                  rsi_14_4h_aligned[i] > 40 and 
                  prices['volume'].iloc[i] > 2.0 * vol_ma_20_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or 4h RSI becomes overbought
            if price < S1_aligned[i] or rsi_14_4h_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above R1 or 4h RSI becomes oversold
            if price > R1_aligned[i] or rsi_14_4h_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_Camarilla_R1S1_Breakout_Volume_RSIFilter_v1"
timeframe = "1h"
leverage = 1.0