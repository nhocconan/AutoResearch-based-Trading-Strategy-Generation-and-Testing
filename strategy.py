#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Ensure enough data for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for 1D ATR (volatility filter) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ATR calculation
        return np.zeros(n)
    
    # Calculate 14-period ATR on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low since no previous close
    tr[0] = tr1[0]
    
    # Calculate ATR using Wilder's smoothing (equivalent to RMA)
    atr_14 = np.zeros_like(tr)
    atr_14[0] = tr[0]
    for i in range(1, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Align 1D ATR to 12h timeframe (with 1-bar delay for completed daily bar)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 50-period EMA on 12h data for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).values
    
    # Volume spike detection: current volume > 2.0 * 20-period average volume
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if data not ready
        if (np.isnan(ema_50[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when volatility is elevated (ATR > 50-period ATR average)
        # Calculate 50-period average of ATR for volatility regime filter
        if i >= 100:  # Need enough data for ATR average
            atr_avg_50 = np.mean(atr_14_aligned[i-50:i])
            volatility_filter = atr_14_aligned[i] > 1.2 * atr_avg_50
        else:
            volatility_filter = True  # Default to true during warmup
        
        if position == 0:
            # Long: Price above EMA50 (uptrend) + volume spike + volatility filter
            if close[i] > ema_50[i] and volume[i] > 2.0 * vol_avg_20[i] and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price below EMA50 (downtrend) + volume spike + volatility filter
            elif close[i] < ema_50[i] and volume[i] > 2.0 * vol_avg_20[i] and volatility_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or volatility collapse
            if position == 1:
                # Exit long: Price closes below EMA50 OR volatility drops significantly
                if close[i] < ema_50[i] or atr_14_aligned[i] < 0.8 * atr_14_aligned[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above EMA50 OR volatility drops significantly
                if close[i] > ema_50[i] or atr_14_aligned[i] < 0.8 * atr_14_aligned[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_EMA50_VolumeSpike_ATRFilter"
timeframe = "12h"
leverage = 1.0