#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h EMA(50) trend filter and 1d ADX(14) regime filter.
# Long when RSI < 30 AND price > 4h EMA50 (uptrend bias) AND 1d ADX < 20 (range regime).
# Short when RSI > 70 AND price < 4h EMA50 (downtrend bias) AND 1d ADX < 20 (range regime).
# Exit when RSI crosses 50 (mean reversion complete) OR ADX > 25 (trend regime - avoid mean reversion in strong trends).
# Uses discrete position size 0.20. Session filter 08-20 UTC to avoid low-liquidity hours.
# Target: 80-120 trades over 4 years (20-30/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 1h Indicators: RSI(14) ===
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # === 4h Indicators: EMA(50) for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d Indicators: ADX(14) for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx_1d.values
    
    # Align 1d ADX to 1h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50, 14 for RSI/ADX)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if outside trading session or any required data is NaN
        if not in_session[i] or \
           (np.isnan(rsi_values[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        rsi_val = rsi_values[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        adx_val = adx_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI crosses above 50 (mean reversion) OR ADX > 25 (trend regime)
            if rsi_val > 50 or adx_val > 25:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI crosses below 50 (mean reversion) OR ADX > 25 (trend regime)
            if rsi_val < 50 or adx_val > 25:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: RSI < 30 (oversold) AND price > 4h EMA50 (uptrend bias) AND ADX < 20 (range regime)
            if rsi_val < 30 and price > ema_50_4h_val and adx_val < 20:
                signals[i] = 0.20
                position = 1
            
            # SHORT: RSI > 70 (overbought) AND price < 4h EMA50 (downtrend bias) AND ADX < 20 (range regime)
            elif rsi_val > 70 and price < ema_50_4h_val and adx_val < 20:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_RSI14_4hEMA50_1dADX20_MeanReversion_V1"
timeframe = "1h"
leverage = 1.0