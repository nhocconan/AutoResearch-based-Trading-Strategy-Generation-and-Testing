#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for indicators
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly RSI(7) for momentum
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    alpha = 1/7
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=alpha, adjust=False, min_periods=7).mean().values
    avg_loss = loss_series.ewm(alpha=alpha, adjust=False, min_periods=7).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    
    # Calculate weekly EMA(21) for trend
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly indicators to 6-hour timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility
        atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
        atr_ma_aligned = align_htf_to_ltf(prices, df_1w, atr_ma)
        if np.isnan(atr_ma_aligned[i]):
            signals[i] = 0.0
            continue
        vol_ok = atr_aligned[i] > (atr_ma_aligned[i] * 0.2)
        
        # Trend filter: price above/below weekly EMA21
        uptrend = close[i] > ema_21_aligned[i]
        downtrend = close[i] < ema_21_aligned[i]
        
        # RSI conditions: extreme levels for mean reversion in strong trend
        rsi_extreme_low = rsi_aligned[i] < 20  # Deep oversold in uptrend
        rsi_extreme_high = rsi_aligned[i] > 80  # Deep overbought in downtrend
        
        # Long conditions: uptrend + deep oversold + volatility ok
        long_condition = uptrend and rsi_extreme_low and vol_ok
        
        # Short conditions: downtrend + deep overbought + volatility ok
        short_condition = downtrend and rsi_extreme_high and vol_ok
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: RSI returns to neutral zone
        elif position == 1 and rsi_aligned[i] > 40:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi_aligned[i] < 60:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyRSI_EMA21_MeanReversion"
timeframe = "6h"
leverage = 1.0