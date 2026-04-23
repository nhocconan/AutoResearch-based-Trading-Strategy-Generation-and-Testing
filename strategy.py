#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d EMA34 trend filter with volume spike confirmation.
Long when Williams %R crosses above -80 (oversold bounce) AND 1d EMA34 is rising AND volume > 1.8x 20-period average.
Short when Williams %R crosses below -20 (overbought rejection) AND 1d EMA34 is falling AND volume > 1.8x 20-period average.
Exit when Williams %R crosses -50 (mean reversion midpoint) or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Targets 12-37 trades/year per symbol (50-150 total over 4 years) by using 1d trend filter to reduce false signals.
Williams %R is effective in ranging markets (common in 2025+ BTC/ETH) and captures mean reversion at extremes.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # EMA slope (rising/falling) - compare current vs 3 periods ago
    ema_slope = np.zeros_like(ema_1d_34_aligned)
    ema_slope[3:] = ema_1d_34_aligned[3:] - ema_1d_34_aligned[:-3]
    
    # Calculate Williams %R (14-period) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Williams %R slope for cross detection
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]  # first bar
    
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
    start_idx = max(100, 34, 20, 14, 3)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_34_aligned[i]) or np.isnan(ema_slope[i]) or 
            np.isnan(williams_r[i]) or np.isnan(williams_r_prev[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
        wr_prev = williams_r_prev[i]
        ema_slope_val = ema_slope[i]
        ema_val = ema_1d_34_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold bounce) AND 1d EMA34 rising AND volume spike
            if (wr > -80 and wr_prev <= -80 and 
                ema_slope_val > 0 and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R crosses below -20 (overbought rejection) AND 1d EMA34 falling AND volume spike
            elif (wr < -20 and wr_prev >= -20 and 
                  ema_slope_val < 0 and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R crosses -50 (mean reversion midpoint)
            if position == 1 and wr < -50 and wr_prev >= -50:
                exit_signal = True
            elif position == -1 and wr > -50 and wr_prev <= -50:
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

name = "6H_WilliamsR_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0