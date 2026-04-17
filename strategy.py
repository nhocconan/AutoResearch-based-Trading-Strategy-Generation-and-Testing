#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour price action filtered by 4-hour momentum and 1-day volatility regime.
# Uses 4h RSI for trend bias and 1d ATR ratio for volatility filtering to avoid chop.
# Entry: 1h price crosses above/below 4h EMA with momentum confirmation.
# Exit: opposite signal or volatility expansion.
# Designed for low trade frequency (15-35/year) to minimize fee drag.
# Works in bull via momentum continuation and in bear via mean-reversion in low volatility.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data for trend bias ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    rsi_period = 14
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 4h EMA for trend
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d data for volatility regime ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).ewm(span=50, min_periods=50).mean().values
    atr_ratio_1d = atr_1d / (atr_ma_1d + 1e-10)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # === 1h RSI for entry timing ===
    delta_1h = np.diff(close, prepend=close[0])
    gain_1h = np.maximum(delta_1h, 0)
    loss_1h = np.maximum(-delta_1h, 0)
    avg_gain_1h = pd.Series(gain_1h).ewm(alpha=1/rsi_period, min_periods=rsi_period).mean().values
    avg_loss_1h = pd.Series(loss_1h).ewm(alpha=1/rsi_period, min_periods=rsi_period).mean().values
    rs_1h = avg_gain_1h / (avg_loss_1h + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs_1h))
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid high volatility regimes
        vol_filter = atr_ratio_1d_aligned[i] < 1.2
        
        # Trend bias from 4h RSI
        bullish_bias = rsi_4h_aligned[i] > 50
        bearish_bias = rsi_4h_aligned[i] < 50
        
        # Entry conditions
        if position == 0:
            # Long: price above 4h EMA, bullish bias, low volatility
            if close[i] > ema_4h_aligned[i] and bullish_bias and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: price below 4h EMA, bearish bias, low volatility
            elif close[i] < ema_4h_aligned[i] and bearish_bias and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long: hold until bearish bias or volatility expansion
            if not bullish_bias or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short: hold until bullish bias or volatility expansion
            if not bearish_bias or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hRSI_1dATR_VolumeFilter"
timeframe = "1h"
leverage = 1.0