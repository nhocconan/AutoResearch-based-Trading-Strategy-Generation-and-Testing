#!/usr/bin/env python3
"""
6h_RSI2_Extreme_Reversal_1dTrendFilter_VolumeSpike_v1
Hypothesis: 6h RSI(2) extreme reversals with 1-day trend filter and volume spike confirmation.
Long when RSI(2) < 5 (extremely oversold) in 1-day uptrend with volume > 2x 20-period average.
Short when RSI(2) > 95 (extremely overbought) in 1-day downtrend with volume > 2x 20-period average.
Uses 1-day EMA50 as trend filter to ensure we only trade mean reversions in the direction of the primary trend.
RSI(2) captures short-term exhaustion moves that tend to reverse quickly, especially when aligned with higher timeframe trend.
Designed for 12-37 trades/year (50-150 over 4 years) by requiring confluence of extreme RSI, 1-day trend, and volume spike.
Works in bull/bear via 1-day trend filter: only takes long reversions in uptrend, short in downtrend.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and typical price
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate RSI(2) on 6h close
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_2 = 100 - (100 / (1 + rs))
    
    # Calculate 20-period volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 2 for RSI, 20 for volume MA)
    start_idx = max(2, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_2[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Extreme RSI reversal conditions with trend filter
        if htf_trend[i] == 1:  # Uptrend on 1d
            # Long reversal from extreme oversold with volume spike
            if rsi_2[i] < 5 and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if RSI returns to neutral (50) or we get overbought signal
            elif position == 1 and (rsi_2[i] > 50 or rsi_2[i] > 70):
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
            # Short reversal from extreme overbought with volume spike
            if rsi_2[i] > 95 and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if RSI returns to neutral (50) or we get oversold signal
            elif position == -1 and (rsi_2[i] < 50 or rsi_2[i] < 30):
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

name = "6h_RSI2_Extreme_Reversal_1dTrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0