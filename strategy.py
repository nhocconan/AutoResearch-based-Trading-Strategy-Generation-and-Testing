#!/usr/bin/env python3
"""
4h_RSI20_Stochastic_1dTrend_Volume
Hypothesis: Use RSI(20) with overbought/oversold levels (80/20) combined with 1d EMA trend filter and volume confirmation. Stochastic(14,3,3) confirms momentum exhaustion. Designed for mean reversion in ranging markets while following higher timeframe trend. Target 20-50 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === RSI(20) on 4h ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Stochastic(14,3,3) on 4h ===
    high = prices['high'].values
    low = prices['low'].values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = np.where((highest_high - lowest_low) != 0, k, 50)
    d = pd.Series(k).rolling(window=3, min_periods=3).mean().values
    d_slow = pd.Series(d).rolling(window=3, min_periods=3).mean().values
    
    # === Volume confirmation: 20-period volume average on 4h ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(d_slow[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_34_1d_aligned[i]
        rsi_val = rsi[i]
        stoch_d = d_slow[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: RSI < 20 (oversold) + Stochastic < 20 + volume spike > 1.5 + price above 1d EMA
            if (rsi_val < 20 and 
                stoch_d < 20 and 
                vol_spike > 1.5 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 80 (overbought) + Stochastic > 80 + volume spike > 1.5 + price below 1d EMA
            elif (rsi_val > 80 and 
                  stoch_d > 80 and 
                  vol_spike > 1.5 and 
                  price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when RSI returns to neutral zone (40-60)
            if position == 1 and rsi_val > 40:
                signals[i] = 0.0
                position = 0
            elif position == -1 and rsi_val < 60:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_RSI20_Stochastic_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0