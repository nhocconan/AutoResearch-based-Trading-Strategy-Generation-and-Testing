#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1w EMA200 trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND 1w EMA200 is rising AND volume > 1.8x 20-period average.
Short when Williams %R > -20 (overbought) AND 1w EMA200 is falling AND volume > 1.8x 20-period average.
Exit when Williams %R crosses above -50 (long) or below -50 (short) or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Targets 12-37 trades/year per symbol (50-150 total over 4 years) by using 1w trend filter to reduce false signals.
Williams %R is effective in both bull and bear markets as it captures mean reversion from extremes,
while the 1w EMA200 filter ensures we trade with the major trend, reducing whipsaws.
"""

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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate Williams %R(14) on 6h timeframe
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (hh14 - close) / (hh14 - ll14)
    # Handle division by zero (when hh14 == ll14)
    williams_r = np.where((hh14 - ll14) == 0, -50, williams_r)
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1w_200_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_200)
    
    # EMA slope (rising/falling) - compare current vs 3 periods ago
    ema_slope = np.zeros_like(ema_1w_200_aligned)
    ema_slope[3:] = ema_1w_200_aligned[3:] - ema_1w_200_aligned[:-3]
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 200, 20, 14, 3)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_1w_200_aligned[i]) or 
            np.isnan(ema_slope[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        wr = williams_r[i]
        ema_slope_val = ema_slope[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND 1w EMA200 rising AND volume spike
            if (wr < -80 and 
                ema_slope_val > 0 and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R > -20 (overbought) AND 1w EMA200 falling AND volume spike
            elif (wr > -20 and 
                  ema_slope_val < 0 and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R crosses above -50 (long) or below -50 (short)
            if position == 1 and wr > -50:
                exit_signal = True
            elif position == -1 and wr < -50:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1wEMA200_Trend_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0