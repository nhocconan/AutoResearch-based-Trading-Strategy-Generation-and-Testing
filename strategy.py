#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for ATR-based volatility regime (once)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily ATR(30) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_30_aligned = align_htf_to_ltf(prices, df_1d, atr_30)
    
    # Daily ATR(5) for volatility spike detection
    atr_5 = pd.Series(tr).rolling(window=5, min_periods=5).mean().values
    atr_5_aligned = align_htf_to_ltf(prices, df_1d, atr_5)
    
    # Load weekly data for trend filter (once)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(atr_30_aligned[i]) or np.isnan(atr_5_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime: only trade when volatility is elevated (ATR5 > 1.5 * ATR30)
        vol_regime = atr_5_aligned[i] > 1.5 * atr_30_aligned[i]
        
        if position == 0 and vol_regime:
            # Long: Close above weekly EMA50 with volatility spike
            if close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below weekly EMA50 with volatility spike
            elif close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Volatility drops below threshold or opposite signal
            if position == 1:
                if not vol_regime or close[i] < ema_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if not vol_regime or close[i] > ema_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_VolatilityRegime_WeeklyEMA50_Trend"
timeframe = "12h"
leverage = 1.0