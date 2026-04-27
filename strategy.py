#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 12h trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; entries occur on reversals from extremes
# with volume confirmation and higher timeframe trend alignment. Works in bull/bear by filtering
# trade direction with 12h EMA trend. Target: 50-150 total trades over 4 years (~12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R (14-period) on 12h data
    williams_r = np.full(len(df_12h), np.nan)
    for i in range(13, len(df_12h)):
        highest_high = np.max(high_12h[i-13:i+1])
        lowest_low = np.min(low_12h[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_12h[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Align Williams %R to 4h timeframe (wait for 12h bar close)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # 12h EMA trend filter (50-period)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: volume > 1.5 x 20-period average (approx 10 hours of 4h bars)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h data (13 bars for Williams %R), EMA (50), volume MA (20)
    start_idx = max(13, 50, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Williams %R levels
        wr = williams_r_aligned[i]
        oversold = wr < -80
        overbought = wr > -20
        
        # Trend filter from 12h EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from oversold with volume and bullish trend
            if wr > -80 and williams_r_aligned[i-1] <= -80 and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: Williams %R crosses below -20 from overbought with volume and bearish trend
            elif wr < -20 and williams_r_aligned[i-1] >= -20 and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R reaches overbought or trend turns bearish
            if wr >= -20 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R reaches oversold or trend turns bullish
            if wr <= -80 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsR_14_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0