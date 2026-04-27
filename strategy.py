# 4h_Polarized_Fractal_Efficiency_PFE_EMA34_Breakout_Volume
# Hypothesis: Polarized Fractal Efficiency (PFE) > 60 indicates strong trending markets.
# Combine with EMA34 trend filter and volume confirmation for high-probability breakouts.
# Works in bull/bear by following the trend direction, avoiding counter-trend trades.
# Target: 20-40 trades/year on 4H timeframe to minimize fee drag.

#!/usr/bin/env python3
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
    
    # Get daily data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        alpha = 2 / (34 + 1)
        ema_34_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    
    # Calculate previous day's OHLC for reference (avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Align daily EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4-period volume average for spike detection (4h x 4 = 16h ~ 2/3 day)
    vol_ma = np.full(n, np.nan)
    vol_period = 4
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Calculate PFE (Polarized Fractal Efficiency) on close prices
    pfe_period = 10
    pfe = np.full(n, np.nan)
    for i in range(pfe_period - 1, n):
        # Net price change (numerator)
        num = close[i] - close[i - pfe_period + 1]
        # Sum of absolute price changes (denominator)
        den = 0.0
        for j in range(i - pfe_period + 2, i + 1):
            den += abs(close[j] - close[j-1])
        if den != 0:
            pfe[i] = (num / den) * 100
        else:
            pfe[i] = 0.0
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(34, vol_period, pfe_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(pfe[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.8x average volume
        vol_filter = vol_ratio > 1.8
        
        # PFE filter: > 60 for strong uptrend, < -60 for strong downtrend
        pfe_long = pfe[i] > 60
        pfe_short = pfe[i] < -60
        
        if position == 0:
            # Long: Strong uptrend (PFE > 60), price above daily EMA34, volume spike
            if pfe_long and price > ema_34_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Strong downtrend (PFE < -60), price below daily EMA34, volume spike
            elif pfe_short and price < ema_34_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Trend weakening (PFE < 40) or price below EMA34
            if pfe[i] < 40 or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Trend weakening (PFE > -40) or price above EMA34
            if pfe[i] > -40 or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Polarized_Fractal_Efficiency_PFE_EMA34_Breakout_Volume"
timeframe = "4h"
leverage = 1.0