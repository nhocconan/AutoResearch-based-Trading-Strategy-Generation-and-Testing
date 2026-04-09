#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels for mean reversion and 1w trend filter
# - Uses 1d HTF for Camarilla pivot levels (H3/L3, H4/L4) as key support/resistance
# - Uses 1w HTF for trend direction via EMA50: price above/below EMA50 determines trend bias
# - Long when price touches L3/L4 in uptrend (price > weekly EMA50) with volume confirmation
# - Short when price touches H3/H4 in downtrend (price < weekly EMA50) with volume confirmation
# - Volume confirmation: current 12h volume > 1.5x 20-period average to filter low-quality signals
# - Fixed position size 0.25 to balance risk and reward
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_1w_camarilla_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla levels use previous day's high, low, close
    # H4 = close + 1.5*(high - low)
    # H3 = close + 1.1*(high - low)
    # L3 = close - 1.1*(high - low)
    # L4 = close - 1.5*(high - low)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First bar will have NaN due to roll, handled by min_periods equivalent
    range_1d = prev_high - prev_low
    H4 = prev_close + 1.5 * range_1d
    H3 = prev_close + 1.1 * range_1d
    L3 = prev_close - 1.1 * range_1d
    L4 = prev_close - 1.5 * range_1d
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF data to 12h timeframe (wait for completed HTF bar)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(H4_aligned[i]) or np.isnan(H3_aligned[i]) or
            np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        # Price proximity to Camarilla levels (within 0.2% for touch)
        proximity_threshold = 0.002  # 0.2%
        touch_H4 = abs(close[i] - H4_aligned[i]) / close[i] < proximity_threshold
        touch_H3 = abs(close[i] - H3_aligned[i]) / close[i] < proximity_threshold
        touch_L3 = abs(close[i] - L3_aligned[i]) / close[i] < proximity_threshold
        touch_L4 = abs(close[i] - L4_aligned[i]) / close[i] < proximity_threshold
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price reaches opposite level (H3/H4) or trend changes
            if touch_H3 or touch_H4 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price reaches opposite level (L3/L4) or trend changes
            if touch_L3 or touch_L4 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic based on trend and Camarilla touch
            if volume_confirmed:
                if uptrend and (touch_L3 or touch_L4):
                    # In uptrend, price touches support: long mean reversion
                    position = 1
                    signals[i] = position_size
                elif downtrend and (touch_H3 or touch_H4):
                    # In downtrend, price touches resistance: short mean reversion
                    position = -1
                    signals[i] = -position_size
    
    return signals