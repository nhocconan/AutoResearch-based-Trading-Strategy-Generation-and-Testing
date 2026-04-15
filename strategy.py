#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) + 1d EMA34 trend + volume spike
# Long when Williams %R crosses above -80 from oversold + 1d EMA34 uptrend + volume > 2.0x 20-period avg
# Short when Williams %R crosses below -20 from overbought + 1d EMA34 downtrend + volume > 2.0x 20-period avg
# Williams %R identifies reversals; EMA34 filters trend; volume spike confirms momentum.
# Designed for 6h timeframe to target 50-150 trades over 4 years (12-37/year).
# Discrete position sizing (0.25) controls drawdown and minimizes fee churn.
# Works in both bull (buy oversold dips) and bear (sell overbought rallies) markets.

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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Williams %R(14) on 6h ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Williams %R cross above -80 (oversold exit) and below -20 (overbought exit)
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    williams_r_cross_above_80 = (williams_r_prev <= -80) & (williams_r > -80)
    williams_r_cross_below_20 = (williams_r_prev >= -20) & (williams_r < -20)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 14, 20) + 5  # EMA34 + WilliamsR(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(williams_r_prev[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (exiting oversold)
        # 2. 1d EMA34 uptrend (close > EMA34)
        # 3. Volume confirmation
        if williams_r_cross_above_80[i] and \
           (close[i] > ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (exiting overbought)
        # 2. 1d EMA34 downtrend (close < EMA34)
        # 3. Volume confirmation
        elif williams_r_cross_below_20[i] and \
             (close[i] < ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR14_1dEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0