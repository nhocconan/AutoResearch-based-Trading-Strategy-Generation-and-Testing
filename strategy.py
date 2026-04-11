#!/usr/bin/env python3
# 12h_1d_momentum_trend_v1
# Strategy: 12h momentum with 1d EMA trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: In trending markets, momentum combined with higher timeframe trend filter
# captures sustainable moves while avoiding counter-trend whipsaws. Volume confirmation
# ensures institutional participation. Designed for low trade frequency (<30/year) to
# minimize fee drag, effective in both bull and bear markets via trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_momentum_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h momentum (10-period ROC)
    mom = np.zeros_like(close)
    mom[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    # 12h RSI (14-period) for overbought/oversold
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[:14])
    avg_loss[14] = np.mean(loss[:14])
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(mom[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or \
           np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: avoid low volatility (range) markets
        if i >= 50:
            atr_ma = pd.Series(atr[:i+1]).rolling(window=50, min_periods=50).mean().iloc[-1]
            vol_filter = atr[i] > 0.8 * atr_ma  # Only trade when volatility is above 80% of average
        else:
            vol_filter = True
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: close vs 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Momentum conditions
        mom_strength = abs(mom[i]) > 1.0  # Minimum 1% momentum
        mom_bullish = mom[i] > 0
        mom_bearish = mom[i] < 0
        
        # RSI filter: avoid extremes
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Entry conditions
        # Long: Bullish momentum AND uptrend AND volume confirmation AND volatility filter
        if mom_bullish and mom_strength and uptrend and vol_confirm and vol_filter and \
           rsi_not_overbought and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Bearish momentum AND downtrend AND volume confirmation AND volatility filter
        elif mom_bearish and mom_strength and downtrend and vol_confirm and vol_filter and \
             rsi_not_oversold and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Momentum reversal or RSI extreme
        elif position == 1 and (mom_bearish or rsi[i] >= 70):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (mom_bullish or rsi[i] <= 30):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals