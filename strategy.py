#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI(14) with 4h ADX(14) trend filter and volume spike
# Long when RSI < 30, 4h ADX > 25, and volume spike (oversold bounce in uptrend)
# Short when RSI > 70, 4h ADX > 25, and volume spike (overbought rejection in uptrend)
# Uses 4h for trend strength and direction (via ADX), 1h for entry timing
# Volume spike confirms momentum; avoids choppy/low-volume false signals
# Targets 60-150 total trades over 4 years (15-37/year) to balance opportunity and cost
# Session filter: 08-20 UTC to avoid low-liquidity Asian session noise

name = "1h_RSI14_4hADX_Volume_Spike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data once for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 4h
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.nansum(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (np.zeros_like(plus_dm))
        minus_di = 100 * (np.zeros_like(minus_dm))
        dx = np.zeros_like(high)
        
        # Smooth +DM and -DM
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1])
        for i in range(period+1, len(plus_dm)):
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        # ADX: smoothed DX
        adx = np.zeros_like(dx)
        adx[2*period] = np.nansum(dx[period:2*period+1]) / period
        for i in range(2*period+1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # RSI(14) on 1h
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1h = calculate_rsi(close, 14)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for RSI and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1h[i]) or np.isnan(adx_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi_1h[i]
        adx_val = adx_4h_aligned[i]
        vol_spike = volume_spike[i]
        price = close[i]
        
        if position == 0:
            # Enter long: RSI < 30 (oversold), ADX > 25 (strong trend), volume spike
            if rsi_val < 30 and adx_val > 25 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Enter short: RSI > 70 (overbought), ADX > 25 (strong trend), volume spike
            elif rsi_val > 70 and adx_val > 25 and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (neutral) or ADX < 20 (weak trend)
            if rsi_val > 50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI < 50 (neutral) or ADX < 20 (weak trend)
            if rsi_val < 50 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals