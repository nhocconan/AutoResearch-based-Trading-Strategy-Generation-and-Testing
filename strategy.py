#!/usr/bin/env python3
# 1d_ema_crossover_volatility_regime_v1
# Hypothesis: Daily EMA crossover (21/50) with volatility regime filter (ATR ratio < 1.2) and volume confirmation.
# In trending markets, EMA crossover captures momentum; low volatility filter avoids whipsaws.
# Volume confirmation ensures institutional participation. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Target: 30-100 total trades over 4 years by requiring EMA cross + vol regime + volume spike.
# Primary timeframe: 1d, HTF: 1w for trend filter (EMA200) to avoid counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ema_crossover_volatility_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily EMA21 and EMA50 for crossover signal
    ema21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # ATR(14) for volatility regime filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (to detect low volatility regimes)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)
    
    # Volume confirmation: 20-period average
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema21[i]) or np.isnan(ema50[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: only trade when ATR ratio < 1.2 (low volatility)
        low_volatility = atr_ratio[i] < 1.2
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Trend filter: only trade in direction of weekly EMA200
        bullish_trend = close[i] > ema200_1w_aligned[i]
        bearish_trend = close[i] < ema200_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: EMA21 crosses below EMA50 or volatility spikes
            if ema21[i] < ema50[i] or not low_volatility:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA21 crosses above EMA50 or volatility spikes
            if ema21[i] > ema50[i] or not low_volatility:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if low_volatility and volume_confirmed:
                # Long entry: EMA21 crosses above EMA50 AND bullish weekly trend
                if ema21[i] > ema50[i] and ema21[i-1] <= ema50[i-1] and bullish_trend:
                    position = 1
                    signals[i] = 0.25
                # Short entry: EMA21 crosses below EMA50 AND bearish weekly trend
                elif ema21[i] < ema50[i] and ema21[i-1] >= ema50[i-1] and bearish_trend:
                    position = -1
                    signals[i] = -0.25
    
    return signals