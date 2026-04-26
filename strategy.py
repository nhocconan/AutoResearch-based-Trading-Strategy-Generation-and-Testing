#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Filtered_v2
Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1-day trend filter and volatility regime filter.
Only trade breakouts in direction of 1-day EMA34 trend when ATR(12)/ATR(48) < 0.8 (low volatility regime).
Avoids whipsaws in high volatility and reduces fee drift. Uses discrete position sizing (0.25).
Target: 12-37 trades/year (50-150 over 4 years) by requiring confluence of breakout, trend, and low vol regime.
Works in bull/bear via 1-day trend filter: only takes long breakouts in uptrend, short in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Camarilla pivot levels from 1d data
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R1_1d = typical_price_1d + (1.1/12) * (df_1d['high'] - df_1d['low'])  # R1 level
    S1_1d = typical_price_1d - (1.1/12) * (df_1d['high'] - df_1d['low'])  # S1 level
    
    # Align Camarilla levels to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d.values)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d.values)
    
    # Volatility regime filter: ATR(12) / ATR(48) < 0.8 (low volatility)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12 = pd.Series(tr).rolling(window=12, min_periods=12).mean().values
    atr_48 = pd.Series(tr).rolling(window=48, min_periods=48).mean().values
    vol_regime = atr_12 / (atr_48 + 1e-10) < 0.8  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 48 for ATR)
    start_idx = max(34, 48)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_12[i]) or np.isnan(atr_48[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volatility regime condition (low volatility only)
        in_low_vol = vol_regime[i]
        
        # Breakout conditions with trend filter and volatility filter
        if htf_trend[i] == 1:  # Uptrend on 1d
            # Long breakout above R1 in low volatility regime
            if close[i] > R1_1d_aligned[i] and in_low_vol:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if price falls below S1 (reversal signal) or volatility increases
            elif position == 1 and (close[i] < S1_1d_aligned[i] or not in_low_vol):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 1d
            # Short breakdown below S1 in low volatility regime
            if close[i] < S1_1d_aligned[i] and in_low_vol:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if price rises above R1 (reversal signal) or volatility increases
            elif position == -1 and (close[i] > R1_1d_aligned[i] or not in_low_vol):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Filtered_v2"
timeframe = "12h"
leverage = 1.0