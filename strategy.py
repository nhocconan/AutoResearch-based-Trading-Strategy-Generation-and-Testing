#!/usr/bin/env python3
"""
6h_Supertrend_RSI_Regime_v1
Hypothesis: Combine Supertrend trend filter with RSI mean reversion on 6h timeframe, using 1d ADX regime filter to avoid whipsaws. Supertrend provides clear trend direction, RSI(14) < 30 or > 70 signals mean-reversion entries in the direction of trend, and ADX > 25 ensures we only trade in trending markets. This approach works in both bull (trend following + pullbacks) and bear (shorting rallies in downtrends) markets. Target: 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Supertrend calculation (ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = np.zeros(n)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend basic upper/lower bands
    hl2 = (high + low) / 2
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    
    # Supertrend final bands and direction
    upper_final = np.zeros(n)
    lower_final = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    upper_final[0] = upper_basic[0]
    lower_final[0] = lower_basic[0]
    supertrend[0] = lower_final[0]
    trend[0] = 1
    
    for i in range(1, n):
        if close[i] <= upper_final[i-1]:
            upper_final[i] = upper_basic[i]
        else:
            upper_final[i] = upper_final[i-1]
            
        if close[i] >= lower_final[i-1]:
            lower_final[i] = lower_basic[i]
        else:
            lower_final[i] = lower_final[i-1]
        
        if supertrend[i-1] == upper_final[i-1] and close[i] <= upper_final[i]:
            supertrend[i] = lower_final[i]
            trend[i] = -1
        elif supertrend[i-1] == lower_final[i-1] and close[i] >= lower_final[i]:
            supertrend[i] = upper_final[i]
            trend[i] = 1
        else:
            supertrend[i] = supertrend[i-1]
            trend[i] = trend[i-1]
    
    # RSI(14)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
    
    for i in range(rsi_period, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d ADX regime filter (trend strength)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation
    adx_period = 14
    tr1_d = high_1d[1:] - low_1d[1:]
    tr2_d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))])
    
    # Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    tr_smooth = np.zeros_like(tr_d)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    tr_smooth[adx_period-1] = np.sum(tr_d[:adx_period])
    plus_dm_smooth[adx_period-1] = np.sum(plus_dm[:adx_period])
    minus_dm_smooth[adx_period-1] = np.sum(minus_dm[:adx_period])
    
    for i in range(adx_period, len(tr_d)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / adx_period) + tr_d[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / adx_period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / adx_period) + minus_dm[i]
    
    # Avoid division by zero
    plus_di = np.divide(plus_dm_smooth, tr_smooth, out=np.zeros_like(plus_dm_smooth), where=tr_smooth!=0) * 100
    minus_di = np.divide(minus_dm_smooth, tr_smooth, out=np.zeros_like(minus_dm_smooth), where=tr_smooth!=0) * 100
    dx = np.divide(np.abs(plus_di - minus_di), (plus_di + minus_di), out=np.zeros_like(plus_di), where=(plus_di + minus_di)!=0) * 100
    
    adx = np.zeros(len(dx))
    if len(dx) >= adx_period:
        adx[adx_period-1] = np.mean(dx[:adx_period])
        for i in range(adx_period, len(dx)):
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, rsi_period, adx_period, atr_period)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i]) or
            np.isnan(supertrend[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        st = supertrend[i]
        rsi_val = rsi[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: uptrend + RSI oversold + volume spike + strong trend (ADX>25)
            if st == lower_final[i] and rsi_val < 30 and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + RSI overbought + volume spike + strong trend (ADX>25)
            elif st == upper_final[i] and rsi_val > 70 and vol_spike and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: trend change or RSI overbought
            if st == upper_final[i] or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: trend change or RSI oversold
            if st == lower_final[i] or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Supertrend_RSI_Regime_v1"
timeframe = "6h"
leverage = 1.0