#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; entries taken on reversal
# from extreme levels with volume confirmation and higher timeframe trend alignment.
# Works in ranging markets (mean reversion) and captures reversals in trends.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:  # Need enough data for Williams %R
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R for each 1d bar (14-period)
    williams_r = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        highest_high = np.max(high_1d[i-14:i+1])
        lowest_low = np.min(low_1d[i-14:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # Neutral when range is zero
    
    # Williams %R signals: oversold < -80, overbought > -20
    williams_r_oversold = williams_r < -80
    williams_r_overbought = williams_r > -20
    
    # Align Williams %R signals to 6h timeframe (wait for 1d close)
    oversold_aligned = align_htf_to_ltf(prices, df_1d, williams_r_oversold.astype(float))
    overbought_aligned = align_htf_to_ltf(prices, df_1d, williams_r_overbought.astype(float))
    
    # 1d EMA trend filter (50-period)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.8 x 20-period average (~5 days of 6h bars)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (15 bars for Williams %R), EMA (50), volume MA (20)
    start_idx = max(15, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(oversold_aligned[i]) or np.isnan(overbought_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.8 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold reversal with volume and bullish trend
            if oversold_aligned[i] > 0.5 and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: Williams %R overbought reversal with volume and bearish trend
            elif overbought_aligned[i] > 0.5 and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or trend turns bearish
            current_williams_r = williams_r[np.searchsorted(df_1d.index.values, 
                                                            prices.index[i], 
                                                            side='right') - 1] if i >= 15 else -50
            if current_williams_r > -50 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or trend turns bullish
            current_williams_r = williams_r[np.searchsorted(df_1d.index.values, 
                                                            prices.index[i], 
                                                            side='right') - 1] if i >= 15 else -50
            if current_williams_r < -50 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsR_Reversal_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0