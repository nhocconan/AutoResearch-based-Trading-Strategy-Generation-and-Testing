#!/usr/bin/env python3
"""
Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation with 1d EMA50 trend filter and ATR-based volatility filter.
Long when price deviates significantly below VWAP in uptrend with low volatility (mean reversion).
Short when price deviates significantly above VWAP in downtrend with low volatility.
Uses 6h timeframe to capture medium-term mean reversion moves within the dominant trend.
ATR filter ensures we only trade when volatility is normal, avoiding choppy markets.
Target: 12-30 trades/year per symbol (50-120 total over 4 years).
Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h ATR(14) for volatility filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # True Range calculation
    tr1 = high_6h[1:] - low_6h[1:]
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_6h, atr_14)
    
    # Calculate 6h VWAP (cumulative typical price * volume / cumulative volume)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Calculate deviation from VWAP in ATR units
    vwap_dev = (close - vwap) / atr_14_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14)  # need EMA50 and ATR14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vwap_dev[i]) or 
            np.isnan(atr_14_aligned[i]) or atr_14_aligned[i] <= 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA50 = uptrend, close < 1d EMA50 = downtrend
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volatility filter: ATR must be within reasonable range (not too high, not too low)
        # Use 50-period ATR MA to normalize current ATR
        if i >= 50:
            atr_ma_50 = np.nanmean(atr_14_aligned[max(0, i-49):i+1])
            vol_filter = (atr_14_aligned[i] > 0.5 * atr_ma_50) and (atr_14_aligned[i] < 2.0 * atr_ma_50)
        else:
            vol_filter = True  # Not enough data for MA, allow trade
        
        if position == 0:
            # Long: Price significantly below VWAP (>0.8 ATR) in uptrend with normal volatility
            if vwap_dev[i] < -0.8 and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price significantly above VWAP (>0.8 ATR) in downtrend with normal volatility
            elif vwap_dev[i] > 0.8 and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to VWAP (within 0.2 ATR) or trend reversal
            exit_signal = False
            if position == 1:
                # Exit long if price returns to VWAP or trend turns down
                if vwap_dev[i] > -0.2 or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Exit short if price returns to VWAP or trend turns up
                if vwap_dev[i] < 0.2 or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_VWAP_MeanReversion_1dEMA50_Trend_ATR_VolFilter"
timeframe = "6h"
leverage = 1.0