#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses weekly pivot context for higher timeframe bias: long only when weekly pivot shows bullish bias
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Camarilla R3/S3 levels provide high-probability reversal/breakout zones
# 1d EMA34 ensures alignment with daily trend
# Volume confirmation (1.5x 20-period average) filters low-quality breakouts
# Weekly pivot bias prevents trading against higher timeframe structure
# Works in bull markets via breakouts with trend alignment and bear markets via mean reversion at R3/S3
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "6h_Camarilla_R3S3_1dEMA34_VolumeSpike_WeeklyBias"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Camarilla levels from prior completed 6h bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # Using prior bar to avoid look-ahead
    prior_close = np.roll(close, 1)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close[0] = close[0]  # avoid NaN on first bar
    prior_high[0] = high[0]
    prior_low[0] = low[0]
    
    rang = prior_high - prior_low
    r3 = prior_close + 1.1 * rang
    s3 = prior_close - 1.1 * rang
    
    # Calculate 1d EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 1w pivot bias (bullish if price > weekly pivot, bearish if <)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
    wp_high = df_1w['high'].values
    wp_low = df_1w['low'].values
    wp_close = df_1w['close'].values
    weekly_pivot = (wp_high + wp_low + wp_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 6h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > 1d EMA34 (bullish bias) 
            #            AND price > weekly pivot (weekly bullish bias) AND volume spike
            if (close[i] > r3[i] and 
                close[i] > ema_34_aligned[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND price < 1d EMA34 (bearish bias)
            #            AND price < weekly pivot (weekly bearish bias) AND volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema_34_aligned[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below S3 OR below 1d EMA34 (trend change)
            if close[i] < s3[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above R3 OR above 1d EMA34 (trend change)
            if close[i] > r3[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals