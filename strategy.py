#!/usr/bin/env python3
"""
4h_ParabolicSAR_Breakout_TopBottom
Hypothesis: Parabolic SAR acts as dynamic support/resistance. Price breaking above/below SAR with trend alignment and volume confirmation captures trend reversals and continuations. Works in both bull and bear markets by using SAR as trailing stop and entry trigger. Target: 20-50 trades/year per symbol.
"""

name = "4h_ParabolicSAR_Breakout_TopBottom"
timeframe = "4h"
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
    
    # Parabolic SAR calculation
    def calculate_sar(high, low, start=0.02, increment=0.02, maximum=0.2):
        n = len(high)
        sar = np.zeros(n)
        trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
        af = np.zeros(n)     # acceleration factor
        ep = np.zeros(n)     # extreme point
        
        # Initialize
        sar[0] = low[0]
        trend[0] = 1
        af[0] = start
        ep[0] = high[0]
        
        for i in range(1, n):
            if trend[i-1] == 1:  # uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                # SAR cannot exceed the lowest low of the past two periods
                sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
                
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + increment, maximum)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
                
                # Trend reversal
                if low[i] < sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]  # SAR becomes prior EP
                    ep[i] = low[i]
                    af[i] = start
                else:
                    trend[i] = 1
            else:  # downtrend
                sar[i] = sar[i-1] + af[i-1] * (sar[i-1] - ep[i-1])
                # SAR cannot be lower than the highest high of the past two periods
                sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
                
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + increment, maximum)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
                
                # Trend reversal
                if high[i] > sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]  # SAR becomes prior EP
                    ep[i] = high[i]
                    af[i] = start
                else:
                    trend[i] = -1
        
        return sar, trend
    
    sar, psar_trend = calculate_sar(high, low)
    
    # 4h trend: EMA50 (additional confirmation)
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close > ema_50
    downtrend_4h = close < ema_50
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        psar = sar[i]
        psar_trend_val = psar_trend[i]
        uptrend = uptrend_4h[i]
        downtrend = downtrend_4h[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price above SAR, SAR in uptrend, 4h uptrend, 1d uptrend filter, volume confirmation
            if close[i] > psar and psar_trend_val == 1 and uptrend and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: price below SAR, SAR in downtrend, 4h downtrend, 1d downtrend filter, volume confirmation
            elif close[i] < psar and psar_trend_val == -1 and downtrend and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price below SAR or SAR flips to downtrend
            if close[i] < psar or psar_trend_val == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price above SAR or SAR flips to uptrend
            if close[i] > psar or psar_trend_val == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals