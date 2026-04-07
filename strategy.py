#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour momentum strategy with 1-day trend filter and volatility-adjusted position sizing
# Uses 4h RSI(14) for momentum signals, 1d EMA(50) for trend direction, and ATR(14) for volatility scaling
# Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag
# Works in bull markets via momentum continuation and in bear markets via counter-trend at extremes

name = "4h_momentum_1d_trend_vol_scaled_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # ATR(14) for volatility scaling
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: scale position size inversely with volatility
        vol_ratio = atr[i] / atr_ma[i] if atr_ma[i] > 0 else 1.0
        vol_scale = np.clip(1.0 / vol_ratio, 0.5, 1.5)  # scale between 0.5 and 1.5
        base_size = 0.25
        
        # Momentum conditions from RSI
        bullish_momentum = rsi[i] > 50 and rsi[i] < 70  # Avoid overbought
        bearish_momentum = rsi[i] < 50 and rsi[i] > 30  # Avoid oversold
        
        # Trend filter from 1d EMA
        above_trend = close[i] > ema_1d_aligned[i]
        below_trend = close[i] < ema_1d_aligned[i]
        
        # Long: bullish momentum AND above daily trend
        if bullish_momentum and above_trend:
            signals[i] = base_size * vol_scale
        # Short: bearish momentum AND below daily trend
        elif bearish_momentum and below_trend:
            signals[i] = -base_size * vol_scale
        else:
            signals[i] = 0.0
    
    return signals