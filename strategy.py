#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 12h EMA(50) trend filter and ATR volatility filter.
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13).
# Enter long when Bull Power > 0 and rising, Bear Power < 0 and rising, price > 12h EMA(50).
# Enter short when Bear Power < 0 and falling, Bull Power < 0 and falling, price < 12h EMA(50).
# Exit when power signals weaken or price crosses 12h EMA(50).
# ATR filter: only trade when ATR(14) > 0.5 * ATR(50) to avoid low volatility periods.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled risk.

name = "6h_elderray12h_ema50_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # EMA(13) for Elder Ray (calculate on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR(14) > 0.5 * ATR(50)
        vol_filter = atr_14[i] > 0.5 * atr_50[i]
        
        if position == 1:  # long position
            # Exit: Elder Ray weakening OR price crosses below EMA50
            if bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif vol_filter:
                signals[i] = 0.25
            else:
                signals[i] = 0.0  # exit due to low vol
                position = 0
        elif position == -1:  # short position
            # Exit: Elder Ray weakening OR price crosses above EMA50
            if bear_power[i] >= 0 or bull_power[i] <= 0 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif vol_filter:
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # exit due to low vol
                position = 0
        else:
            # Look for entries: Elder Ray signals + EMA50 trend + volatility filter
            if vol_filter:
                # Long: Bull Power > 0 and rising, Bear Power < 0 and rising, price > EMA50
                if (bull_power[i] > 0 and i > 50 and bull_power[i] > bull_power[i-1] and
                    bear_power[i] < 0 and i > 50 and bear_power[i] > bear_power[i-1] and
                    close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 and falling, Bull Power < 0 and falling, price < EMA50
                elif (bear_power[i] < 0 and i > 50 and bear_power[i] < bear_power[i-1] and
                      bull_power[i] < 0 and i > 50 and bull_power[i] < bull_power[i-1] and
                      close[i] < ema_50_12h_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals