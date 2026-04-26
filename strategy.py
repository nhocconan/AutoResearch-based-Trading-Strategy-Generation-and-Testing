#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolFilter
Hypothesis: 1h Camarilla pivot R1/S1 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
In strongly trending markets (price > 4h EMA50), long R1 breakout or short S1 breakout with volume > 1.5x 20-period 1d average.
R1/S1 represent tighter breakout levels, increasing trade frequency while 4h trend filter reduces false signals.
Uses 1d volume spike to ensure institutional participation. Designed for 15-37 trades/year (60-150 over 4 years) by requiring confluence of breakout, trend, and volume.
Works in bull/bear via 4h trend filter: only takes long breakouts in uptrend, short in downtrend.
Uses discrete position sizing (0.20) to minimize fee churn.
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
    
    # Load 4h data ONCE before loop for HTF trend
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for HTF trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    htf_trend = np.where(close > ema_50_4h_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Load 1d data ONCE before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d 20-period volume average for spike detection
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20, adjust=False).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Camarilla pivot levels from 1d data
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R1_1d = typical_price_1d + (1.1/12) * (df_1d['high'] - df_1d['low'])  # R1 level
    S1_1d = typical_price_1d - (1.1/12) * (df_1d['high'] - df_1d['low'])  # S1 level
    
    # Align Camarilla levels to 1h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d.values)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 4h EMA, 20 for 1d volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Volume spike condition (1d volume > 1.5x 20-period average)
        volume_spike = df_1d['volume'].values[i // 24] > 1.5 * vol_ma_20_1d_aligned[i] if i // 24 < len(df_1d) else False
        
        # Breakout conditions with trend filter
        if htf_trend[i] == 1:  # Uptrend on 4h
            # Long breakout above R1 with volume spike
            if close[i] > R1_1d_aligned[i] and volume_spike:
                if position != 1:
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.20
            # Exit long if price falls below S1 (reversal signal)
            elif position == 1 and close[i] < S1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
        elif htf_trend[i] == -1:  # Downtrend on 4h
            # Short breakdown below S1 with volume spike
            if close[i] < S1_1d_aligned[i] and volume_spike:
                if position != -1:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = -0.20
            # Exit short if price rises above R1 (reversal signal)
            elif position == -1 and close[i] > R1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0