#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13954_1h_rsi_multitf_v1"
timeframe = "1h"
leverage = 1.0

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and 1d volatility regime filter.
# Long when: RSI < 30 (oversold) + 4h close > 4h EMA(50) (bullish trend) + 1d ATR ratio < 0.8 (low volatility)
# Short when: RSI > 70 (overbought) + 4h close < 4h EMA(50) (bearish trend) + 1d ATR ratio < 0.8 (low volatility)
# Exit when RSI crosses 50 or volatility regime changes (ATR ratio > 1.2)
# Uses 4h/1d for signal direction and regime filter, 1h only for entry timing.
# Target: 80-150 total trades over 4 years (20-38/year) to balance opportunity and fee drag.

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_rsi(close, period):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for EMA trend filter ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    ema_4h = calculate_ema(df_4h['close'].values, 50)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data for volatility regime filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_1d_ma = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_ma)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1h data for RSI and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # RSI(14)
    rsi = calculate_rsi(close, 14)
    
    # ATR(14) for stop loss and volatility regime
    atr_1h = calculate_atr(high, low, close, 14)
    
    # Volatility regime: current ATR / 10-period MA of ATR
    atr_ma = pd.Series(atr_1h).rolling(window=10, min_periods=10).mean().values
    atr_ratio = atr_1h / (atr_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(50, 14, 10) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(atr_1d_ma_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ratio[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade in low volatility (ATR ratio < 0.8)
        vol_regime_low = atr_ratio[i] < 0.8
        vol_regime_high = atr_ratio[i] > 1.2  # exit condition
        
        # 4h trend filter
        bullish_trend = close[i] > ema_4h_aligned[i]
        bearish_trend = close[i] < ema_4h_aligned[i]
        
        # RSI signals
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        # Entry signals
        long_entry = rsi_oversold and bullish_trend and vol_regime_low
        short_entry = rsi_overbought and bearish_trend and vol_regime_low
        
        # Exit signals
        long_exit = rsi_neutral or vol_regime_high or not bullish_trend
        short_exit = rsi_neutral or vol_regime_high or not bearish_trend
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals