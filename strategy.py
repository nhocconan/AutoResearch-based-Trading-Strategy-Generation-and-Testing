#!/usr/bin/env python3
# 1H_4H_1D_Camarilla_R1_S1_Breakout_Trend_Filter
# Hypothesis: On 1h timeframe, use 4h trend and 1d volatility regime to filter trades, entering only when price breaks Camarilla levels from the previous 1h candle with volume confirmation. This reduces noise by requiring alignment across timeframes and avoids overtrading. Target: 15-30 trades/year per symbol (60-120 total over 4 years).

name = "1H_4H_1D_Camarilla_R1_S1_Breakout_Trend_Filter"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h trend: EMA(34)
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_4h = close_4h > ema_34_4h
    
    # Get 1d data for volatility regime (avoid choppy markets)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) for volatility regime
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d volatility regime: low volatility when ATR < 50-day median
    atr_median = pd.Series(atr_14).rolling(window=50, min_periods=50).median().values
    low_vol_regime = atr_14 < atr_median
    
    # Calculate Camarilla levels for 1h: R1, S1 based on previous 1h candle
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    range_1h = high - low
    # Camarilla R1 = close + (range * 1.1/12)
    # Camarilla S1 = close - (range * 1.1/12)
    camarilla_r1 = close + (range_1h * 1.1 / 12)
    camarilla_s1 = close - (range_1h * 1.1 / 12)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align 4h and 1d indicators to 1h
    trend_up_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_up_4h)
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trend_up_4h_aligned[i]) or np.isnan(low_vol_regime_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Camarilla R1 + 4h uptrend + low vol regime + volume confirmation
            if close[i] > camarilla_r1[i] and trend_up_4h_aligned[i] and low_vol_regime_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below Camarilla S1 + 4h downtrend + low vol regime + volume confirmation
            elif close[i] < camarilla_s1[i] and not trend_up_4h_aligned[i] and low_vol_regime_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Camarilla S1 (reversal) or trend changes or volatility increases
            if close[i] < camarilla_s1[i] or not trend_up_4h_aligned[i] or not low_vol_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above Camarilla R1 (reversal) or trend changes or volatility increases
            if close[i] > camarilla_r1[i] or trend_up_4h_aligned[i] or not low_vol_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals