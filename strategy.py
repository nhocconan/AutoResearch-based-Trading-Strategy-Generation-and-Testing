#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day price action with weekly regime filter (ADX) and daily momentum (RSI)
# Long when price > daily EMA(50) + weekly ADX > 25 (trending) + daily RSI > 55
# Short when price < daily EMA(50) + weekly ADX > 25 + daily RSI < 45
# Exit when price crosses EMA(50) in opposite direction
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25
# Uses weekly ADX for trend strength and daily RSI for momentum confirmation
# Target: 50-150 total trades over 4 years (12-38/year)

name = "1d_ema50_weekly_adx_daily_rsi_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily EMA(50) for trend
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly data for ADX (trend strength filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly ADX(14)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range
    tr1 = high_w - low_w
    tr2 = np.abs(high_w - np.roll(close_w, 1))
    tr3 = np.abs(low_w - np.roll(close_w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_w - np.roll(high_w, 1)) > (np.roll(low_w, 1) - low_w), 
                       np.maximum(high_w - np.roll(high_w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_w, 1) - low_w) > (high_w - np.roll(high_w, 1)), 
                        np.maximum(np.roll(low_w, 1) - low_w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr_w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Daily RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema50[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below EMA50
            elif close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above EMA50
            elif close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: EMA50 break with weekly ADX trend filter and daily RSI momentum
            # Trend filter: weekly ADX > 25 (strong trend)
            adx_filter = adx_aligned[i] > 25
            # Momentum filter: daily RSI > 55 for long, < 45 for short
            rsi_filter_long = rsi[i] > 55
            rsi_filter_short = rsi[i] < 45
            
            # Long: price above EMA50 + strong trend + bullish momentum
            if close[i] > ema50[i] and adx_filter and rsi_filter_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price below EMA50 + strong trend + bearish momentum
            elif close[i] < ema50[i] and adx_filter and rsi_filter_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals