#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA34 trend filter + volume confirmation
# Long when Williams %R(14) crosses above -80 (oversold bounce) + price > 1d EMA34 + volume > 1.5x 20-period avg
# Short when Williams %R(14) crosses below -20 (overbought rejection) + price < 1d EMA34 + volume > 1.5x 20-period avg
# Williams %R identifies mean-reversion extremes in bear markets; EMA34 filters for higher timeframe trend alignment
# Volume confirmation ensures breakout validity. Designed for low frequency (12-25/year) to minimize fee drag.
# Works in both bull (buying dips in uptrend) and bear (selling rallies in downtrend) markets.

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: EMA34 for Trend Filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h Indicators: Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Volume filter: current volume > 1.5x 20-period volume SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 34  # EMA34 needs 34 periods
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_sma_20[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (from below) - oversold bounce
        # 2. Price above 1d EMA34 (uptrend filter)
        # 3. Volume confirmation
        williams_r_prev = williams_r[i-1] if i > 0 else -100
        williams_r_cross_up = (williams_r_prev <= -80) and (williams_r[i] > -80)
        if williams_r_cross_up and (close[i] > ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (from above) - overbought rejection
        # 2. Price below 1d EMA34 (downtrend filter)
        # 3. Volume confirmation
        elif (williams_r_prev >= -20) and (williams_r[i] < -20) and (close[i] < ema_34_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR_1dEMA34_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0