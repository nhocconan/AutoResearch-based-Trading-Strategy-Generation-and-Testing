#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA trend filter and volume confirmation.
# Enters long when Williams %R crosses above -80 (oversold reversal) AND price > EMA200 (uptrend) AND volume > 1.5x average.
# Enters short when Williams %R crosses below -20 (overbought reversal) AND price < EMA200 (downtrend) AND volume > 1.5x average.
# Williams %R identifies momentum reversals; EMA200 filters for trend direction; volume adds conviction.
# Designed for low turnover (target: 15-35 trades/year) to avoid fee drag.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 12h Williams %R (14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 1d EMA200
    close_series_1d = pd.Series(close_1d)
    ema200_1d = close_series_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need sufficient data for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema200_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R signals
        wr_above_80 = williams_r[i] > -80
        wr_below_20 = williams_r[i] < -20
        wr_prev_above_80 = williams_r[i-1] > -80 if i > 0 else False
        wr_prev_below_20 = williams_r[i-1] < -20 if i > 0 else False
        
        # Cross above -80 (bullish reversal)
        wr_cross_up = wr_above_80 and not wr_prev_above_80
        # Cross below -20 (bearish reversal)
        wr_cross_down = wr_below_20 and not wr_prev_below_20
        
        # Trend filter: price relative to EMA200
        price_above_ema = close[i] > ema200_12h[i]
        price_below_ema = close[i] < ema200_12h[i]
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND price > EMA200 AND volume
            if wr_cross_up and price_above_ema and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND price < EMA200 AND volume
            elif wr_cross_down and price_below_ema and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses below -50 OR price crosses below EMA200
            wr_below_50 = williams_r[i] < -50
            wr_prev_above_50 = williams_r[i-1] > -50 if i > 0 else False
            wr_cross_down_50 = wr_below_50 and wr_prev_above_50
            price_cross_below_ema = close[i] < ema200_12h[i] and close[i-1] > ema200_12h[i-1]
            
            if wr_cross_down_50 or price_cross_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses above -50 OR price crosses above EMA200
            wr_above_50 = williams_r[i] > -50
            wr_prev_below_50 = williams_r[i-1] < -50 if i > 0 else False
            wr_cross_up_50 = wr_above_50 and wr_prev_below_50
            price_cross_above_ema = close[i] > ema200_12h[i] and close[i-1] < ema200_12h[i-1]
            
            if wr_cross_up_50 or price_cross_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA200_Volume"
timeframe = "12h"
leverage = 1.0