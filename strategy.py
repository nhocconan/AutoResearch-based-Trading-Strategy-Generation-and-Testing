#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Ehlers Fisher Transform with 1w trend filter and volume confirmation
    # Fisher Transform identifies extreme price movements likely to reverse
    # Weekly trend filter ensures we trade with the higher timeframe momentum
    # Volume confirmation filters for institutional participation
    # Works in bull/bear: reversals happen in all markets, trend filter avoids counter-trend traps
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Ehlers Fisher Transform on close prices
    def fishert_transform(price, length=10):
        """Ehlers Fisher Transform"""
        # Normalize price to [-1, 1] range over lookback period
        highest = pd.Series(price).rolling(window=length, min_periods=length).max().values
        lowest = pd.Series(price).rolling(window=length, min_periods=length).min().values
        
        # Avoid division by zero
        diff = highest - lowest
        diff[diff == 0] = 1e-10
        
        # Normalize to [-1, 1]
        normalized = 2 * (price - lowest) / diff - 1
        
        # Clamp to avoid extreme values in inverse hyperbolic tangent
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Apply Fisher Transform: 0.5 * ln((1+x)/(1-x))
        fishert = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth with 2-period EMA
        fishert_smoothed = pd.Series(fishert).ewm(span=2, adjust=False, min_periods=2).mean().values
        
        return fishert_smoothed
    
    # Calculate Fisher Transform
    fishert = fishert_transform(close, length=10)
    
    # Load 1w data for trend filter (primary trend direction)
    df_1w = get_htf_data(prices, '1w')
    # Use 50-period EMA on weekly close for trend
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma20  # Reduced threshold for more signals
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(fishert[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 (oversold recovery) with weekly uptrend and volume
            if fishert[i] > -1.5 and fishert[i-1] <= -1.5 and ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below +1.5 (overbought decline) with weekly downtrend and volume
            elif fishert[i] < 1.5 and fishert[i-1] >= 1.5 and ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Fisher crosses zero (mean reversion) or opposite extreme
            if position == 1:
                if fishert[i] < 0:  # Cross below zero
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if fishert[i] > 0:  # Cross above zero
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ehlers_Fisher_Transform_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0