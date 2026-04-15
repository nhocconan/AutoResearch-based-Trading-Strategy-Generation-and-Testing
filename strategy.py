#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) with 1d EMA(34) trend filter and volume spike confirmation
# Long when Williams %R < -80 (oversold) + price > 1d EMA(34) + volume > 2.0x 20-period avg
# Short when Williams %R > -20 (overbought) + price < 1d EMA(34) + volume > 2.0x 20-period avg
# Williams %R identifies exhaustion points in 6h cycles, EMA(34) on 1d filters for higher-timeframe trend alignment
# Volume spike confirms conviction at reversal points. Designed for low trade frequency (15-25/year).
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).

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
    
    # === 1d Indicator: EMA(34) (trend filter) ===
    close_1d = df_1d['close'].values
    ema_span = 34
    ema_1d = pd.Series(close_1d).ewm(span=ema_span, adjust=False, min_periods=ema_span).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 6h Indicator: Williams %R (14-period) ===
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * ((highest_high - close) / (highest_high - lowest_low)), 
                          -50)  # neutral when range=0
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(lookback, ema_span) + 20  # Williams %R(14) + EMA(34) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R < -80 (oversold)
        # 2. Price > 1d EMA(34) (uptrend filter)
        # 3. Volume spike confirmation
        if (williams_r[i] < -80) and \
           (close[i] > ema_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R > -20 (overbought)
        # 2. Price < 1d EMA(34) (downtrend filter)
        # 3. Volume spike confirmation
        elif (williams_r[i] > -20) and \
             (close[i] < ema_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR14_1dEMA34_VolumeSpike_Filter_v1"
timeframe = "6h"
leverage = 1.0