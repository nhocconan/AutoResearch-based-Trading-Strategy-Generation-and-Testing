#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Triangular Moving Average (TMA) Crossover with Daily Volatility Filter and Volume Spike
# Uses TMA(21) crossover for trend changes, daily ATR-based volatility filter to avoid low-volatility chop,
# and volume spike for confirmation. Designed for 12-37 trades/year to avoid fee drag.
# Works in trending markets via crossover signals and avoids false signals in low volatility.
name = "12h_TMA_Crossover_DailyVol_Filter_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volatility filter (ATR)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Daily ATR(14) for volatility filter
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr14_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_12h = align_htf_to_ltf(prices, df_daily, atr14_daily)
    
    # 12h TMA(21) for trend - Double SMOOTHED SMA
    sma1 = pd.Series(close).rolling(window=21, min_periods=21).mean().values
    tma21 = pd.Series(sma1).rolling(window=21, min_periods=21).mean().values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 41  # max(21+21-1, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tma21[i]) or np.isnan(atr14_12h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr14_12h[i] > np.nanmedian(atr14_12h[max(0, i-100):i+1])
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: TMA turning up with volatility filter and volume spike
            if tma21[i] > tma21[i-1] and vol_filter and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: TMA turning down with volatility filter and volume spike
            elif tma21[i] < tma21[i-1] and vol_filter and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TMA turning down OR volatility drops below average
            if tma21[i] < tma21[i-1] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TMA turning up OR volatility drops below average
            if tma21[i] > tma21[i-1] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals