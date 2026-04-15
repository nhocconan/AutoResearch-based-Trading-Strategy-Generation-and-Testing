#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and volume confirmation
# Long when Williams %R crosses above -80 from below + price > 1d EMA50 + volume > 1.6x 20-period avg
# Short when Williams %R crosses below -20 from above + price < 1d EMA50 + volume > 1.6x 20-period avg
# Uses discrete position sizing (0.25) to limit drawdown and fee drag.
# Williams %R identifies overbought/oversold conditions; EMA50 filters trend direction; volume confirms strength.
# Designed for low trade frequency (~15-30/year) to avoid fee drag while capturing reversals in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: EMA50 ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Williams %R cross signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    williams_r_cross_up = (williams_r_prev <= -80) & (williams_r > -80)  # cross above -80
    williams_r_cross_down = (williams_r_prev >= -20) & (williams_r < -20)  # cross below -20
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 14, 20) + 5  # EMA50 + Williams %R(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_sma_20[i]) or np.isnan(williams_r_cross_up[i]) or 
            np.isnan(williams_r_cross_down[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.6x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.6)
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (oversold reversal)
        # 2. Price > 1d EMA50 (uptrend filter)
        # 3. Volume confirmation
        if williams_r_cross_up[i] and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (overbought reversal)
        # 2. Price < 1d EMA50 (downtrend filter)
        # 3. Volume confirmation
        elif williams_r_cross_down[i] and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsR14_1dEMA50_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0