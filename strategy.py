#/usr/bin/env python3
"""
1h_HTF_Trend_Filter_Entry
Hypothesis: Use 4h ADX and 1d EMA50 for trend direction, enter on 1h pullbacks to EMA21 with volume confirmation.
In strong trends (ADX>25), price pulls back to EMA21 before continuing. Works in bull/bear by following HTF trend.
Target: 15-30 trades/year (60-120 total) to minimize fee drag.
"""

name = "1h_HTF_Trend_Filter_Entry"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtrader_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for ADX trend strength
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX (14) on 4h
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            up = high[i] - high[i-1]
            down = low[i-1] - low[i]
            plus_dm[i] = up if up > down and up > 0 else 0
            minus_dm[i] = down if down > up and down > 0 else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        if len(high) >= period:
            atr[period-1] = np.mean(tr[1:period])
            plus_di[period-1] = np.mean(plus_dm[1:period]) / atr[period-1] * 100
            minus_di[period-1] = np.mean(minus_dm[1:period]) / atr[period-1] * 100
            
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / atr[i] * 100
                minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / atr[i] * 100
        
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        for i in range(2*period-1, len(high)):
            dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        if len(high) >= 2*period:
            adx[2*period-1] = np.mean(dx[period:2*period])
            for i in range(2*period, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 1d data for EMA50 trend direction
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h EMA21 for entry
    ema21 = np.full(n, np.nan)
    if n >= 21:
        ema21[20] = np.mean(close[:21])
        alpha = 2 / (21 + 1)
        for i in range(21, n):
            ema21[i] = alpha * close[i] + (1 - alpha) * ema21[i-1]
    
    # 1h volume confirmation
    vol_ma20 = np.full(n, np.nan)
    if n >= 20:
        vol_ma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_ma20[i] = (vol_ma20[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21)  # Wait for indicators
    
    for i in range(start_idx, n):
        if np.isnan(adx_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(ema21[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filters
        strong_trend = adx_4h_aligned[i] > 25
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions: pullback to EMA21 with volume
        near_ema21 = abs(close[i] - ema21[i]) / ema21[i] < 0.01  # Within 1% of EMA21
        volume_confirm = volume[i] > vol_ma20[i] * 1.5
        
        if position == 0:
            # Long: uptrend, strong trend, pullback to EMA21
            if strong_trend and uptrend and near_ema21 and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: downtrend, strong trend, pullback to EMA21
            elif strong_trend and downtrend and near_ema21 and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: trend breaks or price moves too far from EMA21
            if not (uptrend and strong_trend) or abs(close[i] - ema21[i]) / ema21[i] > 0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: trend breaks or price moves too far from EMA21
            if not (downtrend and strong_trend) or abs(close[i] - ema21[i]) / ema21[i] > 0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

EOF