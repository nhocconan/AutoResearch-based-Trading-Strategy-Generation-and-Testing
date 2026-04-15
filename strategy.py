#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with volume spike and 1d EMA trend filter
# Long when Williams %R < -80 (oversold) + price > 1d EMA34 (uptrend) + volume > 2.0x 20-period avg
# Short when Williams %R > -20 (overbought) + price < 1d EMA34 (downtrend) + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-25/year).
# Williams %R identifies exhaustion points. EMA34 filter ensures we trade with the 1d trend, avoiding counter-trend whipsaws.
# Volume spike confirms institutional participation at turning points. Works in ranging markets (mean reversion) and strong trends (pullback entries).

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
    
    # === 1d Indicator: EMA34 (trend filter) ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R (14-period) on 12h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R < -80 (oversold)
        # 2. Price > 1d EMA34 (uptrend)
        # 3. Volume confirmation
        if (williams_r[i] < -80) and \
           (close[i] > ema_34_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R > -20 (overbought)
        # 2. Price < 1d EMA34 (downtrend)
        # 3. Volume confirmation
        elif (williams_r[i] > -20) and \
             (close[i] < ema_34_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsR_EMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0