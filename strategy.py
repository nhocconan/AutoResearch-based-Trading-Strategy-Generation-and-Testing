#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator (JAWS/TEETH/LIPS) identifies trend direction and strength.
# Combined with 1d EMA trend filter and volume spike to confirm breakouts.
# Works in bull/bear by filtering trade direction with 1d EMA.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price_1d = (high_1d + low_1d) / 2
    
    # JAWS (13-period SMMA, 8 bars ahead)
    jaws_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # TEETH (8-period SMMA, 5 bars ahead)
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # LIPS (5-period SMMA, 3 bars ahead)
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Williams Alligator lines to 12h timeframe (wait for 1d close)
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # 1d EMA trend filter (34-period)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 2.0 x 24-period average (4 days of 12h bars)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (34 bars for EMA), Williams Alligator (13+8=21), volume MA (24)
    start_idx = max(34, 21, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_34_aligned[i]
        bearish_trend = price < ema_34_aligned[i]
        
        # Williams Alligator signals:
        # Bullish: Lips > Teeth > Jaws (green alignment)
        # Bearish: Lips < Teeth < Jaws (red alignment)
        bullish_alligator = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaws_aligned[i])
        bearish_alligator = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaws_aligned[i])
        
        if position == 0:
            # Long: bullish Alligator alignment with volume and bullish trend
            if bullish_alligator and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: bearish Alligator alignment with volume and bearish trend
            elif bearish_alligator and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator turns bearish or trend turns bearish
            if not bullish_alligator or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Alligator turns bullish or trend turns bullish
            if not bearish_alligator or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Williams_Alligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0