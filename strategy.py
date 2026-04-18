#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Squeeze_Play_v2
Hypothesis: Uses 1d Camarilla pivot levels (H3/L3) with volume confirmation and 4h volume-weighted RSI for mean reversion in ranging markets.
Designed to capture reversals at key pivot levels during low volatility squeezes, effective in both bull and bear markets.
Targets 20-35 trades/year by requiring confluence of price at pivot level, volume spike, and oversold/overbought conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: corrected import name

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # H3 = C + 1.25*(H-L), L3 = C - 1.25*(H-L)
    # H2 = C + 1.0*(H-L), L2 = C - 1.0*(H-L)
    # H1 = C + 0.5*(H-L), L1 = C - 0.5*(H-L)
    # Pivot = (H+L+C)/3
    
    # We'll use H3 and L3 as key reversal levels
    H_minus_L = df_1d['high'].values - df_1d['low'].values
    C = df_1d['close'].values
    
    H3 = C + 1.25 * H_minus_L
    L3 = C - 1.25 * H_minus_L
    
    # Align 1d Camarilla levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # 4h RSI(14) for overbought/oversold conditions
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        # Wilder's smoothing
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # 4h volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # RSI needs 14, volume MA needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(rsi_vals[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price at or below L3 with volume spike and RSI oversold (<30)
            if (close[i] <= L3_aligned[i] and vol_spike[i] and rsi_vals[i] < 30):
                signals[i] = 0.25
                position = 1
            # Short: price at or above H3 with volume spike and RSI overbought (>70)
            elif (close[i] >= H3_aligned[i] and vol_spike[i] and rsi_vals[i] > 70):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses above pivot or RSI overbought
            pivot = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
            if (close[i] > pivot_aligned[i] or rsi_vals[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below pivot or RSI oversold
            pivot = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
            if (close[i] < pivot_aligned[i] or rsi_vals[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Camarilla_Pivot_Squeeze_Play_v2"
timeframe = "4h"
leverage = 1.0