#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 12h EMA trend filter
# Williams %R identifies overbought/oversold conditions on 1d timeframe
# Extreme readings (< -90 or > -10) combined with 12h EMA(21) trend direction
# Volume confirmation (current 6h volume > 1.8x 20-period average) filters false signals
# Works in bull/bear: mean reversion from extremes with trend alignment
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "6h_12h_1d_williamsr_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 20 or len(df_12h) < 25:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    
    # Extreme levels: oversold < -90, overbought > -10
    williams_oversold = williams_r < -90
    williams_overbought = williams_r > -10
    
    # Calculate 12h EMA(21) for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_prev = np.roll(ema_12h, 1)  # Previous bar close for trend
    ema_12h_prev[0] = ema_12h[0]  # First bar
    ema_trend_up = ema_12h > ema_12h_prev
    ema_trend_down = ema_12h < ema_12h_prev
    
    # Align HTF indicators to 6h timeframe
    oversold_aligned = align_htf_to_ltf(prices, df_1d, williams_oversold.astype(float))
    overbought_aligned = align_htf_to_ltf(prices, df_1d, williams_overbought.astype(float))
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_down.astype(float))
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(oversold_aligned[i]) or np.isnan(overbought_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.8x average 6h volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on overbought condition or trend reversal
            if overbought_aligned[i] > 0.5 or not trend_up_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on oversold condition or trend reversal
            if oversold_aligned[i] > 0.5 or not trend_down_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion from extremes with trend alignment and volume
            if volume_confirmed:
                # Long: oversold + uptrend
                if oversold_aligned[i] > 0.5 and trend_up_aligned[i] > 0.5:
                    position = 1
                    signals[i] = 0.25
                # Short: overbought + downtrend
                elif overbought_aligned[i] > 0.5 and trend_down_aligned[i] > 0.5:
                    position = -1
                    signals[i] = -0.25
    
    return signals