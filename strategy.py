#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h ADX(14) + volume spike + 1d trend filter
    # Long: ADX > 25 (trending) + +DI > -DI (bullish momentum) + volume > 2x 20-period avg + 1d close > 1d EMA50
    # Short: ADX > 25 + -DI > +DI (bearish momentum) + volume > 2x 20-period avg + 1d close < 1d EMA50
    # Uses discrete sizing (0.25) to minimize fee drag and volatility-based exit
    # Target: 12-37 trades/year to stay within 6h optimal range (50-150 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ADX and DI using Wilder's smoothing
    # True Range
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First bar
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])  # Seed
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_6h = wilder_smooth(tr, 14)
    plus_di_6h = 100 * wilder_smooth(up_move, 14) / (atr_6h + 1e-10)
    minus_di_6h = 100 * wilder_smooth(down_move, 14) / (atr_6h + 1e-10)
    dx_6h = 100 * np.abs(plus_di_6h - minus_di_6h) / (plus_di_6h + minus_di_6h + 1e-10)
    adx_6h = wilder_smooth(dx_6h, 14)
    
    # Calculate volume average for confirmation (using 6h data)
    vol_avg_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(adx_6h[i]) or 
            np.isnan(plus_di_6h[i]) or
            np.isnan(minus_di_6h[i]) or
            np.isnan(vol_avg_20_6h[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_avg_20_6h[i]
        
        # Trend filter: 6h close above/below EMA50 (from 1d)
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # ADX conditions: strong trend with directional bias
        strong_trend = adx_6h[i] > 25
        bullish_momentum = plus_di_6h[i] > minus_di_6h[i]
        bearish_momentum = minus_di_6h[i] > plus_di_6h[i]
        
        # Entry conditions
        enter_long = strong_trend and bullish_momentum and volume_confirmed and uptrend
        enter_short = strong_trend and bearish_momentum and volume_confirmed and downtrend
        
        # Exit conditions: trend weakening or reversal
        exit_long = position == 1 and (adx_6h[i] < 20 or minus_di_6h[i] > plus_di_6h[i])
        exit_short = position == -1 and (adx_6h[i] < 20 or plus_di_6h[i] > minus_di_6h[i])
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "6h_1d_adx_volume_trend_v1"
timeframe = "6h"
leverage = 1.0