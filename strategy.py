#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA200 trend filter and volume confirmation
# Long when Williams %R crosses above -80 from below + volume > 1.5x 20-period avg + price > 1d EMA200
# Short when Williams %R crosses below -20 from above + volume > 1.5x 20-period avg + price < 1d EMA200
# Williams %R identifies overbought/oversold conditions; EMA200 provides major trend filter
# Volume confirmation reduces false signals. Designed for low frequency (15-35 trades/year)
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend)

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
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === 1d Indicators: EMA200 ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 4h Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_sma_20[i]) or np.isnan(highest_high_14[i]) or 
            np.isnan(lowest_low_14[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R cross conditions
        williams_r_prev = williams_r[i-1]
        williams_r_curr = williams_r[i]
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 from below
        # 2. Volume confirmation
        # 3. Price above 1d EMA200 (uptrend filter)
        if (williams_r_prev <= -80 and williams_r_curr > -80) and vol_confirm and (close[i] > ema_200_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 from above
        # 2. Volume confirmation
        # 3. Price below 1d EMA200 (downtrend filter)
        elif (williams_r_prev >= -20 and williams_r_curr < -20) and vol_confirm and (close[i] < ema_200_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_WilliamsR_1dEMA200_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0