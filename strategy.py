#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Long when Williams %R crosses above -20 from below during bullish 12h trend with volume > 1.2x 20-period average.
# Short when Williams %R crosses below -80 from above during bearish 12h trend with volume confirmation.
# Williams %R identifies overbought/oversold conditions; 12h trend filter ensures trades align with higher timeframe momentum.
# Volume confirmation reduces false signals. Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_williamsr_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # 12h trend filter: bullish/bearish based on close vs open
    df_12h = get_htf_data(prices, '12h')
    open_12h = df_12h['open'].values
    close_12h = df_12h['close'].values
    trend_bullish = close_12h > open_12h  # True for bullish 12h bar
    trend_bearish = close_12h < open_12h   # True for bearish 12h bar
    trend_bullish_aligned = align_htf_to_ltf(prices, df_12h, trend_bullish)
    trend_bearish_aligned = align_htf_to_ltf(prices, df_12h, trend_bearish)
    
    # Volume filter: current volume > 1.2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if Williams %R or trend data not available
        if np.isnan(williams_r[i]) or np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.2
        
        # Williams %R cross signals
        wr_above_80 = williams_r[i] > -80 and (i == 14 or williams_r[i-1] <= -80)
        wr_below_20 = williams_r[i] < -20 and (i == 14 or williams_r[i-1] >= -20)
        
        # Check exits
        if position == 1:  # long position
            # Exit: Williams %R drops below -80 or trend turns bearish
            if (williams_r[i] < -80 or 
                trend_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R rises above -20 or trend turns bullish
            if (williams_r[i] > -20 or 
                trend_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: Williams %R crosses above -80 from below during bullish trend
                if wr_above_80 and trend_bullish_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from above during bearish trend
                elif wr_below_20 and trend_bearish_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals