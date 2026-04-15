#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R extreme reversal with 1w EMA trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) + price > 1w EMA34 (bullish trend) + volume > 1.5x 20-period avg
# Short when Williams %R > -20 (overbought) + price < 1w EMA34 (bearish trend) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (10-25/year).
# Williams %R identifies exhaustion points; 1w EMA ensures we trade with the higher timeframe trend.
# Volume confirmation avoids false reversals. Works in ranging markets (mean reversion) and trends (pullbacks).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: Williams %R (14-period) ===
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicator: EMA34 (trend filter) ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: 20-period SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R < -80 (oversold)
        # 2. Price > 1w EMA34 (bullish higher timeframe trend)
        # 3. Volume confirmation (> 1.5x 20-period average)
        if (williams_r_aligned[i] < -80) and \
           (close[i] > ema_34_1w_aligned[i]) and \
           (volume[i] > (vol_sma_20[i] * 1.5)):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R > -20 (overbought)
        # 2. Price < 1w EMA34 (bearish higher timeframe trend)
        # 3. Volume confirmation (> 1.5x 20-period average)
        elif (williams_r_aligned[i] > -20) and \
             (close[i] < ema_34_1w_aligned[i]) and \
             (volume[i] > (vol_sma_20[i] * 1.5)):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_WilliamsR_EMA34_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0