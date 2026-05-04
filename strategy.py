#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation
# Uses Jaw/Teeth/Lips crossover for entry, 1d EMA34 for higher-timeframe trend alignment,
# and volume > 2.0 x 20-period EMA for confirmation. Discrete position sizing (0.25) to
# minimize fee churn. Alligator is effective in ranging markets (common in 2025 BTC/ETH),
# while EMA34 filter ensures we only trade in the direction of the daily trend.
# Target: 20-35 trades/year per symbol.

name = "4h_WilliamsAlligator_1dEMA34_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data for Williams Alligator (SMAs of median price)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Williams Alligator lines:
    # Jaw (Blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (Red): 8-period SMMA, shifted 5 bars ahead
    # Lips (Green): 5-period SMMA, shifted 3 bars ahead
    # Note: Using SMA as approximation for SMMA (simplified)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines (already shifted, so no additional delay needed)
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Get 4h data for volume EMA(20) for volume confirmation
    vol_4h = df_4h['volume'].values
    vol_ema_20 = pd.Series(vol_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        # 1d trend: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = close[i] > ema_34_1d_aligned[i]
        bearish_trend = close[i] < ema_34_1d_aligned[i]
        
        # Williams Alligator signals:
        # Bullish: Lips > Teeth > Jaw (green above red above blue)
        # Bearish: Lips < Teeth < Jaw (green below red below blue)
        bullish_alligator = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alligator = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Long: Bullish Alligator alignment + volume confirmation + bullish 1d trend
            if bullish_alligator and volume_confirmed and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + volume confirmation + bearish 1d trend
            elif bearish_alligator and volume_confirmed and bearish_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish Alligator alignment OR 1d trend turns bearish
            if bearish_alligator or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish Alligator alignment OR 1d trend turns bullish
            if bullish_alligator or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals