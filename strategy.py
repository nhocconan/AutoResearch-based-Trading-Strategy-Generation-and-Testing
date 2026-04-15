#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) with 1d EMA (34) trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold) + price > 1d EMA34 + volume > 1.5x 20-period avg
# Short when Williams %R crosses below -20 (overbought) + price < 1d EMA34 + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year.
# Williams %R identifies mean-reversion extremes in ranging markets. EMA34 filter ensures we only
# take mean-reversion trades in the direction of the daily trend. Works in bull markets (buy dips),
# bear markets (sell rallies), and ranging markets by requiring EMA alignment.

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: EMA (34) ===
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # === 6h Indicator: Williams %R (14) ===
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close) / (highest_high - lowest_low),
                          -50)  # fallback when range=0
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(lookback, 34, 20) + 5
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (from below)
        # 2. Price > 1d EMA34 (bullish trend filter)
        # 3. Volume confirmation
        if (williams_r[i] > -80 and williams_r[i-1] <= -80) and \
           (close[i] > ema_34_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (from above)
        # 2. Price < 1d EMA34 (bearish trend filter)
        # 3. Volume confirmation
        elif (williams_r[i] < -20 and williams_r[i-1] >= -20) and \
             (close[i] < ema_34_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR14_1dEMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0