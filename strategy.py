#!/usr/bin/env python3
# Hypothesis: 12h Williams %R with 1d trend filter and volume confirmation.
# Williams %R measures momentum: values below -80 = oversold, above -20 = overbought.
# Long when Williams %R crosses above -80 from below (bullish reversal) AND price > 1d EMA50 AND volume > 1.3x average.
# Short when Williams %R crosses below -20 from above (bearish reversal) AND price < 1d EMA50 AND volume > 1.3x average.
# Exit when Williams %R reaches opposite extreme (-20 for long, -80 for short) OR trend reversal.
# Uses 12h timeframe for lower frequency, Williams %R for momentum timing, 1d EMA for trend filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via buying dips in uptrend, bear via selling rallies in downtrend.

name = "12h_WilliamsR_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high_12h).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_12h).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    
    # Volume filter: current 12h volume > 1.3x 20-period average
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = volume_12h > (1.3 * vol_ma_12h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below (bullish reversal) 
            # AND price > 1d EMA50 AND volume confirmation
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema50_1d_aligned[i] and volume_filter_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from above (bearish reversal)
            # AND price < 1d EMA50 AND volume confirmation
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema50_1d_aligned[i] and volume_filter_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R reaches -20 (overbought) OR trend reversal (price < 1d EMA50)
            if williams_r[i] >= -20 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R reaches -80 (oversold) OR trend reversal (price > 1d EMA50)
            if williams_r[i] <= -80 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals