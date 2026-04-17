#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R with 1d/1w Trend Filter and Volume Spike.
Long when Williams %R < -80 (oversold) AND price > 1d EMA50 (bullish trend) AND volume > 1.5x 20-period average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA50 (bearish trend) AND volume > 1.5x 20-period average.
Exit when Williams %R crosses above -50 for longs or below -50 for shorts, or trend reverses.
Uses 1d EMA50 for trend filter and 1w EMA200 for higher-timeframe trend confirmation.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R captures mean reversion extremes,
while higher-timeframe EMAs filter for trend alignment to reduce false signals in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w data for EMA200 super trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for super trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Williams %R on 12h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Calculate volume spike filter (volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Align HTF indicators to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        price = close[i]
        ema50 = ema50_1d_aligned[i]
        ema200 = ema200_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA50 (bullish trend) 
            #        AND price > 1w EMA200 (super bullish trend) AND volume spike
            if wr < -80 and price > ema50 and price > ema200 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA50 (bearish trend)
            #         AND price < 1w EMA200 (super bearish trend) AND volume spike
            elif wr > -20 and price < ema50 and price < ema200 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR price < 1d EMA50 (trend reversal)
            if wr > -50 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR price > 1d EMA50 (trend reversal)
            if wr < -50 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_VolumeSpike_1dEMA50_1wEMA200_Trend"
timeframe = "12h"
leverage = 1.0