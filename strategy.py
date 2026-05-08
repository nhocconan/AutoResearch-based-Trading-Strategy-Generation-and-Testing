#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Liquidity_Imbalance_Reversal_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once for trend filter and liquidity imbalance detection
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h close for trend filter (EMA34)
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_12h = (close_12h > ema34_12h).astype(float)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # 12h volume for volume spike detection
    volume_12h = df_12h['volume'].values
    vol_ma12 = pd.Series(volume_12h).rolling(window=12, min_periods=12).mean().values
    vol_spike_12h = volume_12h > (vol_ma12 * 1.8)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Calculate 6h liquidity imbalance: look for rapid price rejection at swing points
    # Using 6-period RSI on 6h to detect overextension
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(span=6, adjust=False, min_periods=6).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume-weighted price change for momentum exhaustion
    price_change = np.diff(close, prepend=close[0])
    vol_weighted_change = price_change * volume
    vol_weighted_ma = pd.Series(vol_weighted_change).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_12h_aligned[i]) or np.isnan(vol_spike_12h_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_weighted_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: RSI oversold + negative volume-weighted momentum (selling exhaustion) + 12h uptrend
            long_setup = (rsi[i] < 30 and vol_weighted_ma[i] < 0 and trend_12h_aligned[i] > 0.5)
            # Short setup: RSI overbought + positive volume-weighted momentum (buying exhaustion) + 12h downtrend
            short_setup = (rsi[i] > 70 and vol_weighted_ma[i] > 0 and trend_12h_aligned[i] < 0.5)
            
            # Require volume spike on 12h for confirmation
            if long_setup and vol_spike_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif short_setup and vol_spike_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or momentum shifts positive
            if rsi[i] > 70 or vol_weighted_ma[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold or momentum shifts negative
            if rsi[i] < 30 or vol_weighted_ma[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h liquidity imbalance reversal strategy.
# Uses 6h RSI to detect overextension and volume-weighted momentum to identify exhaustion.
# 12h trend filter ensures we trade with higher timeframe momentum.
# 12h volume spike confirms institutional participation.
# Works in both bull/bear markets by fading exhaustion spikes in trending environments.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.