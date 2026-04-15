#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA200 trend filter and volume spike confirmation
# Long when Williams %R(14) crosses above -80 from below (oversold bounce) + volume > 2.0x 20-period avg + price > 1d EMA200
# Short when Williams %R(14) crosses below -20 from above (overbought rejection) + volume > 2.0x 20-period avg + price < 1d EMA200
# Williams %R identifies exhaustion points in both bull and bear markets, while 1d EMA200 ensures alignment with higher timeframe trend
# Volume spike confirms institutional participation at turning points. Designed for low frequency (15-25/year) to minimize fee drag.
# Works in bull markets by buying oversold dips in uptrends, and in bear markets by selling overbought rallies in downtrends.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicators: EMA200 ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Primary 6h Indicators: Williams %R(14) ===
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Williams %R signals: cross above -80 (long), cross below -20 (short)
    williams_long_signal = (williams_r > -80) & (np.roll(williams_r, 1) <= -80)
    williams_short_signal = (williams_r < -20) & (np.roll(williams_r, 1) >= -20)
    
    # Volume filter: current volume > 2.0x 20-period volume SMA
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_sma_20 * 2.0)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(williams_long_signal[i]) or
            np.isnan(williams_short_signal[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (oversold bounce)
        # 2. Volume confirmation (institutional participation)
        # 3. Price above 1d EMA200 (uptrend alignment)
        if williams_long_signal[i] and vol_confirm[i] and (close[i] > ema_200_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (overbought rejection)
        # 2. Volume confirmation (institutional participation)
        # 3. Price below 1d EMA200 (downtrend alignment)
        elif williams_short_signal[i] and vol_confirm[i] and (close[i] < ema_200_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR_1dEMA200_Volume_Spike_Filter_v1"
timeframe = "6h"
leverage = 1.0