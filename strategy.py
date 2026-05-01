#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H4/L4 mean reversion with 1d EMA34 trend filter and volume spike confirmation.
# Uses Camarilla H4/L4 levels for mean reversion entries, filtered by 1d EMA34 trend (counter-trend) and volume > 1.8x 20-period median.
# Works in ranging markets (buy at H4, sell at L4) and trending markets (fade extremes with volume confirmation).
# Discrete position sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

name = "4h_Camarilla_H4L4_MeanReversion_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Camarilla levels from previous day OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla H4 and L4 levels (mean reversion zones)
    camarilla_h4 = prev_close + (prev_high - prev_low) * 1.125
    camarilla_l4 = prev_close - (prev_high - prev_low) * 1.125
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA34 and volume median
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 - we mean revert when price is extended from trend
        price_above_ema = curr_close > ema_34_1d_aligned[i]
        price_below_ema = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.8x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.8)
        
        # Camarilla mean reversion conditions (H4/L4 for fading extremes)
        fade_up = curr_close < camarilla_h4_aligned[i]   # price falls below H4 (sell signal)
        fade_down = curr_close > camarilla_l4_aligned[i] # price rises above L4 (buy signal)
        
        if position == 0:  # Flat - look for new entries
            # Long: Price above L4 AND below EMA34 AND volume confirmation (fade downtrend extreme)
            if fade_down and price_below_ema and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price below H4 AND above EMA34 AND volume confirmation (fade uptrend extreme)
            elif fade_up and price_above_ema and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla H4 touch (mean reversion target)
            if curr_close >= camarilla_h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Camarilla L4 touch (mean reversion target)
            if curr_close <= camarilla_l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals