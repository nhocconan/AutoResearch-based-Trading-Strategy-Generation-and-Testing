#!/usr/bin/env python3
"""
6h_WeeklyPivot_VWAP_Reversal
6h strategy using weekly VWAP and monthly pivot points with reversal logic.
- Long: Price pulls back to weekly VWAP during monthly uptrend + RSI < 40
- Short: Price rallies to weekly VWAP during monthly downtrend + RSI > 60
- Exit: Opposite signal or RSI crosses 50
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (buy pullbacks to VWAP) and bear markets (sell rallies to VWAP)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get monthly data for pivot points and trend
    df_1m = get_htf_data(prices, '1M')
    
    # Calculate monthly pivot points (using previous month's data)
    high_1m = df_1m['high'].values
    low_1m = df_1m['low'].values
    close_1m = df_1m['close'].values
    
    # Monthly pivot = (H + L + C) / 3
    pivot_1m = (high_1m + low_1m + close_1m) / 3.0
    # R1 = 2*P - L
    r1_1m = 2 * pivot_1m - low_1m
    # S1 = 2*P - H
    s1_1m = 2 * pivot_1m - high_1m
    
    # Monthly trend: EMA50 vs EMA200
    ema_50_1m = pd.Series(close_1m).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1m = pd.Series(close_1m).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get weekly data for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Typical price for VWAP
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    # VWAP = sum(price * volume) / sum(volume)
    pv_1w = typical_price_1w * volume_1w
    cum_pv_1w = np.nancumsum(pv_1w)
    cum_vol_1w = np.nancumsum(volume_1w)
    vwap_1w = np.divide(cum_pv_1w, cum_vol_1w, out=np.full_like(cum_pv_1w, np.nan), where=cum_vol_1w!=0)
    
    # RSI(14) on 6h close
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 0.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align all data to 6h timeframe
    pivot_1m_aligned = align_htf_to_ltf(prices, df_1m, pivot_1m)
    r1_1m_aligned = align_htf_to_ltf(prices, df_1m, r1_1m)
    s1_1m_aligned = align_htf_to_ltf(prices, df_1m, s1_1m)
    ema_50_1m_aligned = align_htf_to_ltf(prices, df_1m, ema_50_1m)
    ema_200_1m_aligned = align_htf_to_ltf(prices, df_1m, ema_200_1m)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1m_aligned[i]) or np.isnan(r1_1m_aligned[i]) or 
            np.isnan(s1_1m_aligned[i]) or np.isnan(ema_50_1m_aligned[i]) or
            np.isnan(ema_200_1m_aligned[i]) or np.isnan(vwap_1w_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Monthly trend conditions
        uptrend = ema_50_1m_aligned[i] > ema_200_1m_aligned[i]
        downtrend = ema_50_1m_aligned[i] < ema_200_1m_aligned[i]
        
        # Distance to weekly VWAP (normalized by price)
        dist_to_vwap = (close[i] - vwap_1w_aligned[i]) / vwap_1w_aligned[i]
        
        if position == 0:
            # Long: monthly uptrend + pullback to VWAP + RSI oversold
            if uptrend and dist_to_vwap > -0.02 and dist_to_vwap < 0.02 and rsi[i] < 40:
                signals[i] = 0.25
                position = 1
            # Short: monthly downtrend + rally to VWAP + RSI overbought
            elif downtrend and dist_to_vwap > -0.02 and dist_to_vwap < 0.02 and rsi[i] > 60:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or RSI > 50
            if not uptrend or rsi[i] > 50:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or RSI < 50
            if not downtrend or rsi[i] < 50:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_VWAP_Reversal"
timeframe = "6h"
leverage = 1.0