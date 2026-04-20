#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily ADX (14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[np.isnan(adx)] = 0
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate weekly EMA (21) for long-term trend
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 12h RSI (14) for momentum
    close_12h = prices['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isnan(rsi)] = 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Get values
        close_val = prices['close'].iloc[i]
        adx_val = adx_aligned[i]
        ema_trend_val = ema_21_1w_aligned[i]
        rsi_val = rsi[i]
        
        # Skip if any value is invalid
        if (np.isnan(adx_val) or np.isnan(ema_trend_val) or 
            np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong weekly uptrend + moderate ADX + RSI not overbought
            if (close_val > ema_trend_val and 
                adx_val > 20 and adx_val < 40 and 
                rsi_val < 60):
                signals[i] = 0.25
                position = 1
            # Short: Strong weekly downtrend + moderate ADX + RSI not oversold
            elif (close_val < ema_trend_val and 
                  adx_val > 20 and adx_val < 40 and 
                  rsi_val > 40):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly trend turns down OR RSI overbought OR ADX weak
            if (close_val < ema_trend_val or 
                rsi_val > 70 or 
                adx_val < 15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend turns up OR RSI oversold OR ADX weak
            if (close_val > ema_trend_val or 
                rsi_val < 30 or 
                adx_val < 15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_WeeklyTrend_ADX_RSI_Filter_V1
# Uses weekly EMA(21) for long-term trend direction
# Enters long when price above weekly EMA with ADX 20-40 and RSI < 60
# Enters short when price below weekly EMA with ADX 20-40 and RSI > 40
# Exits on trend reversal, RSI extremes, or weak ADX (<15)
# Designed for 12h timeframe with ~15-30 trades/year
name = "12h_WeeklyTrend_ADX_RSI_Filter_V1"
timeframe = "12h"
leverage = 1.0