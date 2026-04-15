#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d EMA34 trend filter and volume spike
# Long when Williams %R(14) < -80 (oversold) + volume > 2.0x 20-period avg + price > 1d EMA34
# Short when Williams %R(14) > -20 (overbought) + volume > 2.0x 20-period avg + price < 1d EMA34
# Williams %R identifies extreme reversals; volume confirms conviction; 1d EMA34 filters trend direction
# Designed for low trade frequency (12-25/year) to minimize fee drag while capturing mean reversion
# Works in both bull and bear markets by requiring volume confirmation and trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d Indicators: EMA34 ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 12h Williams %R (14-period) ===
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 40
    
    for i in range(warmup, n):
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R < -80 (oversold)
        # 2. Volume confirmation
        # 3. Price above 1d EMA34 (uptrend filter)
        if (williams_r[i] < -80) and vol_confirm and (close[i] > ema_34_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R > -20 (overbought)
        # 2. Volume confirmation
        # 3. Price below 1d EMA34 (downtrend filter)
        elif (williams_r[i] > -20) and vol_confirm and (close[i] < ema_34_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsR_Volume_1dEMA34_Filter_v1"
timeframe = "12h"
leverage = 1.0