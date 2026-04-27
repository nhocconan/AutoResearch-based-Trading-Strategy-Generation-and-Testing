#!/usr/bin/env python3
"""
12h_ParabolicSAR_Trend_Momentum
Hypothesis: Parabolic SAR identifies trend direction and potential reversals. 
Combined with RSI momentum filter (40-60 range) to avoid chop, and volume confirmation.
Uses 1d ADX as trend strength filter to avoid weak trends. Designed for 12h timeframe 
to target 15-35 trades per year, works in bull via SAR below price + bullish momentum, 
bear via SAR above price + bearish momentum. Low-frequency signals reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    period = 14
    tr_sum = pd.Series(tr_1d).rolling(window=period, min_periods=period).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=period, min_periods=period).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=period, min_periods=period).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Parabolic SAR (0.02 step, 0.2 max)
    psar = np.zeros(n)
    psar[0] = low[0]
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    ep = high[0] if trend == 1 else low[0]  # extreme point
    
    for i in range(1, n):
        if trend == 1:  # uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if psar[i] > low[i]:  # trend reversal
                trend = -1
                psar[i] = high[i-1]
                af = 0.02
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:  # downtrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if psar[i] < high[i]:  # trend reversal
                trend = 1
                psar[i] = low[i-1]
                af = 0.02
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align with original index
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = max(30, 20)  # ADX/RSI/vol need ~30 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(psar[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_1d_aligned[i]
        rsi_val = rsi[i]
        psar_val = psar[i]
        vol_confirm_val = vol_confirm[i]
        
        # Trend strength filter: ADX > 20
        if adx_val < 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: SAR below price AND RSI > 40 AND RSI < 60 AND volume confirmation
            if psar_val < close[i] and 40 < rsi_val < 60 and vol_confirm_val:
                signals[i] = size
                position = 1
            # Short: SAR above price AND RSI > 40 AND RSI < 60 AND volume confirmation
            elif psar_val > close[i] and 40 < rsi_val < 60 and vol_confirm_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: SAR above price (trend reversal) OR RSI > 70 (overbought)
            if psar_val > close[i] or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: SAR below price (trend reversal) OR RSI < 30 (oversold)
            if psar_val < close[i] or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_ParabolicSAR_Trend_Momentum"
timeframe = "12h"
leverage = 1.0