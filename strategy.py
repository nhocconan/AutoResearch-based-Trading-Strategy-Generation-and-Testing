# 1d_1w_RSI_WeakTrend_MeanReversion
# Hypothesis: On daily timeframe, RSI extremes in weak trends (ADX < 25) indicate mean reversion opportunities.
# Uses weekly trend filter (ADX > 25) to avoid counter-trend trades in strong trends.
# Weekly ADX ensures we only trade mean reversion when higher timeframe is not strongly trending.
# Daily RSI < 30 for long, > 70 for short with volume confirmation.
# Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
# Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RSI_WeakTrend_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ADX for trend strength filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original length
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        up_move = np.concatenate([[np.nan], up_move])
        down_move = np.concatenate([[np.nan], down_move])
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values using Wilder's smoothing (alpha = 1/period)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period + 1:
                return result
            # First value is simple average
            result[period] = np.nanmean(data[1:period+1])
            # Subsequent values: Wilder's smoothing
            for i in range(period + 1, len(data)):
                result[i] = (result[i-1] * (period - 1) + data[i]) / period
            return result
        
        atr = wilders_smoothing(tr, period)
        plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
        
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                      0.0)
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Daily RSI for mean reversion signals
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.insert(delta, 0, np.nan)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period + 1:
                return result
            result[period] = np.nanmean(data[1:period+1])
            for i in range(period + 1, len(data)):
                result[i] = (result[i-1] * (period - 1) + data[i]) / period
            return result
        
        avg_up = wilders_smoothing(up, period)
        avg_down = wilders_smoothing(down, period)
        rs = np.divide(avg_up, avg_down, out=np.full_like(avg_up, np.nan), where=avg_down!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close, 14)
    
    # Volume filter: current volume > 1.3x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(adx_1w_aligned[i]) or np.isnan(rsi_1d[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Only trade when weekly trend is weak (ADX < 25) to avoid strong trends
        weak_trend = adx_1w_aligned[i] < 25
        volume_ok = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: RSI oversold in weak trend with volume
            if rsi_1d[i] < 30 and weak_trend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in weak trend with volume
            elif rsi_1d[i] > 70 and weak_trend and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI returns to neutral (50) or trend strengthens
            if rsi_1d[i] >= 50 or adx_1w_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI returns to neutral (50) or trend strengthens
            if rsi_1d[i] <= 50 or adx_1w_aligned[i] >= 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals