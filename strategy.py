#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot breakout with volume confirmation and ADX regime filter
# Long when price breaks above 1d Camarilla R3 level AND volume > 1.5 * avg_volume(20) AND 12h ADX > 25
# Short when price breaks below 1d Camarilla S3 level AND volume > 1.5 * avg_volume(20) AND 12h ADX > 25
# Exit when price crosses 12h EMA50 (trend reversal) OR ADX < 20 (range regime)
# Uses discrete sizing 0.30 to balance return and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Camarilla provides clear structure with proven breakout edge
# Volume confirmation filters weak breakouts
# 12h ADX regime filter ensures we only trade in trending markets (works in bull/bear)

name = "12h_1dCamarillaR3S3_Volume_ADX_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for pivots
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels based on previous 1d bar
    # Camarilla: R3 = Close + 1.1 * (High - Low), S3 = Close - 1.1 * (High - Low)
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d)
    
    # Get 12h data ONCE before loop for EMA50 and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50 and ADX
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend exit
    close_series_12h = pd.Series(close_12h)
    ema_50_12h = close_series_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h ADX for regime filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Align 1d Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Align 12h EMA50 and ADX to 12h timeframe (already aligned, but use for consistency)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with volume confirmation and ADX > 25 (trending)
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                volume_confirm[i] and adx_12h_aligned[i] > 25):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 1d Camarilla S3 with volume confirmation and ADX > 25 (trending)
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  volume_confirm[i] and adx_12h_aligned[i] > 25):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA50 OR ADX < 20 (range regime)
            if close[i] < ema_50_12h_aligned[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above 12h EMA50 OR ADX < 20 (range regime)
            if close[i] > ema_50_12h_aligned[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals