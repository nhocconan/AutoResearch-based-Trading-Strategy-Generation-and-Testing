#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h ADX for trend strength and 1d RSI for momentum, with price > 4h EMA50 for long bias and < 4h EMA50 for short bias. 
# In strong trends (ADX > 25), price tends to continue in the direction of the EMA50 trend. 
# Uses 1h RSI for entry timing: long when RSI < 30 (oversold pullback in uptrend), short when RSI > 70 (overbought pullback in downtrend). 
# Exit when trend weakens (ADX < 20) or RSI reverts to neutral (40-60 range). 
# Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years (15-37/year) with size 0.20.

name = "1h_ADX_EMA_RSI_Pullback"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h ADX for trend strength
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    close_4h = df_4h['close']
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_4h = wilders_smooth(tr, 14)
    plus_di_4h = 100 * wilders_smooth(plus_dm, 14) / atr_4h
    minus_di_4h = 100 * wilders_smooth(minus_dm, 14) / atr_4h
    dx_4h = 100 * np.abs(plus_di_4h - minus_di_4h) / (plus_di_4h + minus_di_4h)
    adx_4h = wilders_smooth(dx_4h, 14)
    
    # 4h EMA50 for trend direction
    ema_50_4h = close_4h.ewm(span=50, adjust=False).mean().values
    
    # 1d RSI for momentum
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    
    # 1h RSI for entry timing
    delta_h = pd.Series(close).diff()
    gain_h = delta_h.where(delta_h > 0, 0)
    loss_h = -delta_h.where(delta_h < 0, 0)
    avg_gain_h = gain_h.ewm(com=13, adjust=False).mean()
    avg_loss_h = loss_h.ewm(com=13, adjust=False).mean()
    rs_h = avg_gain_h / avg_loss_h
    rsi_1h = 100 - (100 / (1 + rs_h))
    
    # Align HTF indicators to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: strong uptrend (ADX > 25), price above EMA50, RSI oversold (<30)
            if (adx_4h_aligned[i] > 25 and 
                close[i] > ema_50_4h_aligned[i] and 
                rsi_1h[i] < 30 and 
                rsi_1d_aligned[i] > 50):  # Additional bullish bias from daily RSI
                signals[i] = 0.20
                position = 1
            # Enter short: strong downtrend (ADX > 25), price below EMA50, RSI overbought (>70)
            elif (adx_4h_aligned[i] > 25 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  rsi_1h[i] > 70 and 
                  rsi_1d_aligned[i] < 50):  # Additional bearish bias from daily RSI
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend weakens (ADX < 20) or RSI reverts to neutral (>=50) or price below EMA50
            if (adx_4h_aligned[i] < 20 or 
                rsi_1h[i] >= 50 or 
                close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend weakens (ADX < 20) or RSI reverts to neutral (<=50) or price above EMA50
            if (adx_4h_aligned[i] < 20 or 
                rsi_1h[i] <= 50 or 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals