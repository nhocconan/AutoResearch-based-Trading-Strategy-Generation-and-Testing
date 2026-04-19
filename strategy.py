#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Volume + Volume Acceleration
# Long when ADX > 25 (strong trend), volume > 1.5x 20-period average, and volume acceleration > 0 (increasing volume)
# Short when ADX > 25, volume > 1.5x 20-period average, and volume acceleration < 0 (decreasing volume) AND price < 6h SMA50
# Uses daily trend filter: only long when price > daily EMA50, only short when price < daily EMA50
# Designed for 6h timeframe to capture trending moves with volume confirmation in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "6h_ADX_Volume_Acceleration"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros_like(close)
        valid = (plus_di[period:] + minus_di[period:]) > 0
        dx[period:] = np.where(valid, 100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:]), 0)
        
        adx = np.zeros_like(close)
        if len(dx) >= 2*period+1:
            adx[2*period] = np.mean(dx[period:2*period+1])
            for i in range(2*period+1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume and volume acceleration
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_change = np.diff(volume, prepend=volume[0])
    vol_accel = np.diff(vol_change, prepend=vol_change[0])
    
    # Get 1d data for daily EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure ADX, volume MA, and daily EMA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        vol_accel_val = vol_accel[i]
        ema_50_val = ema_50_aligned[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # ADX trend strength filter
        strong_trend = adx_val > 25
        
        if position == 0:
            # Enter long if price above daily EMA50, strong trend, volume confirmation, and increasing volume
            if price > ema_50_val and strong_trend and volume_confirmed and vol_accel_val > 0:
                signals[i] = 0.25
                position = 1
            # Enter short if price below daily EMA50, strong trend, volume confirmation, decreasing volume
            elif price < ema_50_val and strong_trend and volume_confirmed and vol_accel_val < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below daily EMA50 or trend weakens
            if price < ema_50_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above daily EMA50 or trend weakens
            if price > ema_50_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals