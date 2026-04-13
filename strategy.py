#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend-following with 4h ADX filter and 1d RSI mean-reversion
# Uses 4h ADX to confirm trending markets (ADX > 25) and 1d RSI for entry timing
# Long when: 4h ADX > 25 + 1h close > 1h EMA20 + 1d RSI < 40 (pullback in uptrend)
# Short when: 4h ADX > 25 + 1h close < 1h EMA20 + 1d RSI > 60 (bounce in downtrend)
# Position size: 0.20 to manage drawdown, max 40-60 trades/year target
# Session filter: 08-20 UTC to avoid low-volume periods

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for ADX calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h ADX Calculation ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h[0] = high_4h[0] - low_4h[0]
    
    # Directional Movement
    up_move = np.diff(high_4h, prepend=high_4h[0])
    down_move = np.diff(low_4h, prepend=low_4h[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    atr_4h = wilders_smoothing(tr_4h, period_adx)
    plus_di_4h = wilders_smoothing(plus_dm, period_adx) * 100 / np.where(atr_4h == 0, 1, atr_4h)
    minus_di_4h = wilders_smoothing(minus_dm, period_adx) * 100 / np.where(atr_4h == 0, 1, atr_4h)
    dx_4h = np.abs(plus_di_4h - minus_di_4h) / np.where((plus_di_4h + minus_di_4h) == 0, 1, (plus_di_4h + minus_di_4h)) * 100
    adx_4h = wilders_smoothing(dx_4h, period_adx)
    
    # === 1d RSI Calculation ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(len(gain)):
        if i < 14:
            avg_gain[i] = np.mean(gain[:i+1]) if i > 0 else gain[i]
            avg_loss[i] = np.mean(loss[:i+1]) if i > 0 else loss[i]
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # === 1h EMA20 ===
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20
    
    for i in range(20, n):  # Start after EMA20 warmup
        # Skip if data not ready
        if (np.isnan(adx_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session, flatten position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Trend strength filter: ADX > 25
        strong_trend = adx_4h_aligned[i] > 25
        
        # Entry conditions
        uptrend = close[i] > ema_20[i]
        downtrend = close[i] < ema_20[i]
        
        # Long: uptrend + pullback (RSI < 40)
        long_entry = strong_trend and uptrend and (rsi_1d_aligned[i] < 40)
        # Short: downtrend + bounce (RSI > 60)
        short_entry = strong_trend and downtrend and (rsi_1d_aligned[i] > 60)
        
        # Exit conditions: trend weakening or RSI extreme reversal
        exit_long = position == 1 and (not strong_trend or not uptrend or rsi_1d_aligned[i] > 70)
        exit_short = position == -1 and (not strong_trend or not downtrend or rsi_1d_aligned[i] < 30)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_adx_1d_rsi_trend_pullback_v1"
timeframe = "1h"
leverage = 1.0