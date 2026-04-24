#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h ADX(14) regime filter and volume spike confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for ADX trend strength and 1d for higher-timeframe bias.
- RSI(14) < 30 = oversold (long), RSI(14) > 70 = overbought (short) on 1h.
- Regime filter: Only take longs when 4h ADX < 25 (range market), shorts when 4h ADX < 25.
  In strong trends (ADX >= 25), avoid mean reversion to prevent whipsaw.
- Volume confirmation: current 1h volume > 1.5x 20-period volume MA to ensure participation.
- Higher-timeframe bias: 1d close > 1d EMA50 for long bias, < for short bias.
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Works in bull via buying oversold dips in uptrend bias, in bear via selling overbought rallies in downtrend bias.
- Avoids false signals in strong trends via ADX filter, reducing fee-damaging whipsaw trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filtering
    # prices['open_time'] is datetime64[ns], so .dt.hour works
    hours = pd.to_datetime(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for ADX regime filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX(14) on 4h
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # Align with original length
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
        def WilderSmooth(data, period):
            smoothed = np.full_like(data, np.nan)
            if len(data) < period:
                return smoothed
            # First value is simple average
            smoothed[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed[t] = (smoothed[t-1] * (period-1) + data[t]) / period
            for i in range(period, len(data)):
                smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
            return smoothed
        
        tr_smooth = WilderSmooth(tr, period)
        plus_dm_smooth = WilderSmooth(plus_dm, period)
        minus_dm_smooth = WilderSmooth(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = WilderSmooth(dx, period)
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Get 1d data for higher-timeframe bias (close vs EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        gain = np.concatenate([[0.0], gain])
        loss = np.concatenate([[0.0], loss])
        
        # Wilder's smoothing
        def WilderSmooth(data, period):
            smoothed = np.full_like(data, np.nan)
            if len(data) < period:
                return smoothed
            smoothed[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
            return smoothed
        
        avg_gain = WilderSmooth(gain, period)
        avg_loss = WilderSmooth(loss, period)
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1h = calculate_rsi(close, 14)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(rsi_1h[i]) or np.isnan(adx_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine higher-timeframe bias
        long_bias = close[i] > ema_50_1d_aligned[i]
        short_bias = close[i] < ema_50_1d_aligned[i]
        
        # Regime filter: only trade in range markets (ADX < 25)
        in_range = adx_4h_aligned[i] < 25
        
        if position == 0 and in_range:
            # Long: oversold RSI with long bias and volume spike
            if rsi_1h[i] < 30 and long_bias and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: overbought RSI with short bias and volume spike
            elif rsi_1h[i] > 70 and short_bias and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral (50) or opposite signal
            if rsi_1h[i] >= 50 or (rsi_1h[i] > 70 and short_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI returns to neutral (50) or opposite signal
            if rsi_1h[i] <= 50 or (rsi_1h[i] < 30 and long_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_ADX14_Range_VolumeSpike_EMA50Bias_v1"
timeframe = "1h"
leverage = 1.0