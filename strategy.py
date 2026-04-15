#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# Long when price > Alligator Jaw (13-period SMMA) + 1w close > 1w EMA34 + volume > 1.5x 20-period avg
# Short when price < Alligator Jaw (13-period SMMA) + 1w close < 1w EMA34 + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (10-25/year).
# Williams Alligator identifies trend initiation/continuation. Weekly EMA34 filter ensures alignment with higher timeframe trend.
# Works in bull markets (price above jaw with weekly uptrend) and bear markets (price below jaw with weekly downtrend).

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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # === 1w Indicator: EMA34 (trend filter) ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d Indicator: Williams Alligator (Jaw: 13-period SMMA of median price) ===
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(13, 34, 20) + 5  # Alligator(13) + EMA34(34) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price > Alligator Jaw (13-period SMMA)
        # 2. Weekly trend up (1w close > 1w EMA34)
        # 3. Volume confirmation
        if (close[i] > jaw[i]) and \
           (close[i] > ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price < Alligator Jaw (13-period SMMA)
        # 2. Weekly trend down (1w close < 1w EMA34)
        # 3. Volume confirmation
        elif (close[i] < jaw[i]) and \
             (close[i] < ema_34_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_WilliamsAlligator_Jaw_1wEMA34_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0