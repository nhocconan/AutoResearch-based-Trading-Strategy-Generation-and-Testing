#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) + 1d EMA34 trend + volume confirmation
# Long when Williams %R crosses above -80 (oversold bounce) + price > 1d EMA34 + volume > 1.5x 20-period avg
# Short when Williams %R crosses below -20 (overbought rejection) + price < 1d EMA34 + volume > 1.5x 20-period avg
# Williams %R captures mean reversion swings within the trend, effective in both bull and bear markets.
# Volume filter ensures participation, reducing false signals. Target: ~50-100 trades/year on 6h.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Primary Indicator: Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Range: -100 to 0, oversold < -80, overbought > -20
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    williams_r = np.where(hh_ll != 0, (highest_high - close) / hh_ll * -100, -50)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(14, 34, 20) + 5  # Williams %R(14) + EMA34 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Williams %R cross above -80 (oversold bounce)
        williams_cross_up = (williams_r[i] > -80) and (williams_r[i-1] <= -80)
        # Williams %R cross below -20 (overbought rejection)
        williams_cross_down = (williams_r[i] < -20) and (williams_r[i-1] >= -20)
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (oversold bounce)
        # 2. Price > 1d EMA34 (uptrend filter)
        # 3. Volume confirmation
        if williams_cross_up and \
           (close[i] > ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (overbought rejection)
        # 2. Price < 1d EMA34 (downtrend filter)
        # 3. Volume confirmation
        elif williams_cross_down and \
             (close[i] < ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR14_1dEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0