#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) in bull trend (close > 1d EMA50) with volume > 1.8x 24-period MA.
# Short when Alligator jaws > teeth > lips (bearish alignment) in bear trend (close < 1d EMA50) with volume spike.
# Uses 12h primary timeframe with 1d HTF for trend filter and Williams Alligator.
# Williams Alligator (SMAs with 5,8,13 periods) provides trend direction and strength.
# 1d EMA50 filters counter-trend whipsaw. Volume confirmation reduces false breakouts.
# Discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_WilliamsAlligator_1dEMA50_Volume"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw (blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (red): 8-period SMMA, shifted 5 bars ahead
    # Lips (green): 5-period SMMA, shifted 3 bars ahead
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted values with NaN for invalid periods
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align Alligator lines to ensure no look-ahead (use completed 12h bar values)
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, prices, lips_shifted)
    
    # Volume regime: current 12h volume > 1.8x 24-period MA
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Alligator alignment
        is_bullish_alligator = jaw_val < teeth_val < lips_val
        is_bearish_alligator = jaw_val > teeth_val > lips_val
        
        # Entry logic
        if position == 0:
            if is_bull_trend and is_bullish_alligator and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and is_bearish_alligator and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator loses bullish alignment OR trend reversal
            if not is_bullish_alligator or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator loses bearish alignment OR trend reversal
            if not is_bearish_alligator or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals