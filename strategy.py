#!/usr/bin/env python3
"""
6h_AntiTrend_Reversion
Hypothesis: Mean reversion after extreme momentum moves using 6h RSI(2) and 1d ATR filter.
In both bull and bear markets, price tends to revert after sharp moves. RSI(2) < 10 signals
oversold (long), > 90 signals overbought (short). Filter trades to only occur when price is
near the 1d VWAP (mean) and volatility is elevated (ATR > 1.5x ATR(30)) to catch reversals
after volatility spikes. Position size 0.25 targets 20-40 trades/year.
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
    
    # Get 1d data for VWAP and ATR filters
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 6h RSI(2)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    rsi_period = 2
    
    # Wilder's smoothing
    if len(gain) >= rsi_period:
        avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
        for i in range(rsi_period + 1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100  # all gains
    rsi[avg_gain == 0] = 0    # all losses
    
    # 1d VWAP calculation
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vwap_num = np.cumsum(typical_price_1d * df_1d['volume'].values)
    vwap_den = np.cumsum(df_1d['volume'].values)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # 1d ATR(30) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]
    
    atr_period = 30
    atr = np.zeros_like(tr)
    if len(tr) >= atr_period:
        atr[atr_period] = np.mean(tr[1:atr_period+1])
        for i in range(atr_period + 1, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Align 1d indicators to 6h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)  # RSI is already 6h but aligned for safety
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Additional filters: price near VWAP (within 0.5%) and high volatility
    price_pct_from_vwap = np.abs((close - vwap_aligned) / vwap_aligned) * 100
    vwap_filter = price_pct_from_vwap < 0.5
    vol_filter = atr_aligned > (1.5 * np.roll(atr_aligned, 30))  # ATR > 1.5x past 30d ATR
    
    signals = np.zeros(n)
    
    start_idx = max(30, 50)  # need enough data for ATR and VWAP
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_aligned[i]) or np.isnan(vwap_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vwap_filter[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        if rsi_aligned[i] < 10 and vwap_filter[i] and vol_filter[i]:
            # Oversold + near VWAP + high volatility -> long
            signals[i] = 0.25
        elif rsi_aligned[i] > 90 and vwap_filter[i] and vol_filter[i]:
            # Overbought + near VWAP + high volatility -> short
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_AntiTrend_Reversion"
timeframe = "6h"
leverage = 1.0