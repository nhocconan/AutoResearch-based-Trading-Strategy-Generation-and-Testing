#!/usr/bin/env python3
"""
4h_HTF_RSI2_Stochastic_Crossover_TrendFilter_Volume
Hypothesis: RSI(2) + Stochastic(14,3,3) cross with 1d EMA50 trend filter and volume confirmation.
This captures mean-reversion bounces in strong trends across bull/bear markets by using 1d trend
for direction and fast RSI/Stoch for precise entry timing. Volume spike filters false signals.
Designed for 4h timeframe with ~25-40 trades/year to avoid fee drag.
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
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === RSI(2) on 4h ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # === Stochastic(14,3,3) on 4h ===
    high = prices['high'].values
    low = prices['low'].values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    stoch_k = np.where(highest_high - lowest_low != 0, 
                       100 * (close - lowest_low) / (highest_high - lowest_low), 50)
    stoch_k_series = pd.Series(stoch_k)
    stoch_k_smooth = stoch_k_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    stoch_d = pd.Series(stoch_k_smooth).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(stoch_k_smooth[i]) or
            np.isnan(stoch_d[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_1d = ema_50_1d_aligned[i]
        rsi_val = rsi[i]
        stoch_k_val = stoch_k_smooth[i]
        stoch_d_val = stoch_d[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: RSI2 < 10 + Stoch K crosses above D + volume spike > 1.3 + price above 1d EMA50
            if (rsi_val < 10 and 
                stoch_k_val > stoch_d_val and 
                stoch_k_smooth[i-1] <= stoch_d[i-1] and
                vol_spike > 1.3 and 
                price_close > trend_1d):
                signals[i] = 0.25
                position = 1
            # Short: RSI2 > 90 + Stoch K crosses below D + volume spike > 1.3 + price below 1d EMA50
            elif (rsi_val > 90 and 
                  stoch_k_val < stoch_d_val and 
                  stoch_k_smooth[i-1] >= stoch_d[i-1] and
                  vol_spike > 1.3 and 
                  price_close < trend_1d):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when RSI2 crosses 50 in opposite direction
            if position == 1 and rsi_val < 50 and rsi[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            elif position == -1 and rsi_val > 50 and rsi[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_RSI20_Stochastic_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0