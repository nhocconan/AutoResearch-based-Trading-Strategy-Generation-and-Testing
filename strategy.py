#!/usr/bin/env python3
"""
1h_RSI_MeanReversion_4hTrendFilter_VolumeSpike
Hypothesis: On 1h timeframe, use RSI(14) for mean reversion entries (long when RSI<30, short when RSI>70) 
filtered by 4h EMA50 trend (long only when price>EMA50, short only when price<EMA50) and volume spike confirmation.
Session filter (08-20 UTC) reduces noise. Designed to work in both bull and bear markets by fading extremes 
in the direction of the higher timeframe trend. Target: 15-37 trades/year on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 4h EMA50 (50), RSI (14), volume MA (20)
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI<30 (oversold) + 4h uptrend + volume spike
            long_setup = (rsi_values[i] < 30) and (close[i] > ema_50_4h_aligned[i]) and volume_spike[i]
            # Short: RSI>70 (overbought) + 4h downtrend + volume spike
            short_setup = (rsi_values[i] > 70) and (close[i] < ema_50_4h_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: RSI>50 (mean reversion) OR 4h trend turns down
            if (rsi_values[i] > 50) or (close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: RSI<50 (mean reversion) OR 4h trend turns up
            if (rsi_values[i] < 50) or (close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_RSI_MeanReversion_4hTrendFilter_VolumeSpike"
timeframe = "1h"
leverage = 1.0