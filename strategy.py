#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h RSI for direction, 4h ADX for trend strength, and 1h VWAP for entry timing.
# Long when 4h RSI > 50 and 4h ADX > 25 and 1h price > VWAP. Short when 4h RSI < 50 and 4h ADX > 25 and 1h price < VWAP.
# Uses 4h for signal direction and trend strength, 1h only for VWAP entry timing to reduce false signals.
# Includes session filter (08-20 UTC) to avoid low-volume periods. Fixed size 0.20 to limit risk.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.

name = "1h_VWAP_4hRSI_ADX"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI and ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h RSI(14)
    delta = np.diff(close_4h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_4h)
    avg_loss = np.zeros_like(close_4h)
    avg_gain[14] = np.mean(gain[:14])
    avg_loss[14] = np.mean(loss[:14])
    for i in range(15, len(close_4h)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h = np.concatenate([[np.nan], rsi_4h])
    
    # 4h ADX(14)
    def calculate_adx(high, low, close, period=14):
        tr = np.zeros_like(close)
        plus_dm = np.zeros_like(close)
        minus_dm = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
        
        atr = np.zeros_like(close)
        atr[period] = np.mean(tr[:period+1])
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(close)
        minus_di = np.zeros_like(close)
        dx = np.zeros_like(close)
        for i in range(period, len(close)):
            if atr[i] != 0:
                plus_di[i] = 100 * np.mean(plus_dm[i-period+1:i+1]) / atr[i]
                minus_di[i] = 100 * np.mean(minus_dm[i-period+1:i+1]) / atr[i]
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
                else:
                    dx[i] = 0
        
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    
    # Align 4h indicators to 1h
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 1h VWAP
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, 0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(adx_4h_aligned[i]) or np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h RSI > 50, 4h ADX > 25, 1h price > VWAP
            if rsi_4h_aligned[i] > 50 and adx_4h_aligned[i] > 25 and close[i] > vwap[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h RSI < 50, 4h ADX > 25, 1h price < VWAP
            elif rsi_4h_aligned[i] < 50 and adx_4h_aligned[i] > 25 and close[i] < vwap[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: opposite conditions or ADX weak
            if rsi_4h_aligned[i] < 50 or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: opposite conditions or ADX weak
            if rsi_4h_aligned[i] > 50 or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals