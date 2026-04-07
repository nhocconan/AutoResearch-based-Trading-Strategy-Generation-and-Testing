#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray Index + 1-week Trend + Volume Spike
# Elder Ray measures bull/bear power via EMA(13) difference from high/low.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling (bullish divergence)
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising (bearish divergence)
# 1-week EMA(50) filter ensures alignment with higher timeframe trend
# Volume spike (>1.5x 20-period average) confirms institutional participation
# Works in both bull and bear markets by measuring intrinsic strength/weakness
# Target: 15-35 trades/year (60-140 over 4 years)
name = "6h_elder_ray_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Elder Ray Index: EMA(13) of close
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean()
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema13.values
    # Bear Power = EMA(13) - Low
    bear_power = ema13.values - low
    
    # Slope of Bull/Bear Power (3-period change)
    bull_power_slope = pd.Series(bull_power).diff(3)
    bear_power_slope = pd.Series(bear_power).diff(3)
    
    # 1-week EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean()
    weekly_ema_6h = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume filter: current volume > 1.5x 20-period average (volume spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_slope[i]) or np.isnan(bear_power_slope[i]) or 
            np.isnan(weekly_ema_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power turns positive (bulls losing control) or volume spike fails
            if bear_power[i] > 0 or not vol_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Bull Power turns negative (bears losing control) or volume spike fails
            if bull_power[i] < 0 or not vol_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume spike for institutional validation
            if vol_spike[i]:
                # Long: Bull Power positive AND rising (bulls in control and gaining strength)
                #        Bear Power negative AND falling (bears weak and weakening)
                if (bull_power[i] > 0 and bull_power_slope[i] > 0 and 
                    bear_power[i] < 0 and bear_power_slope[i] < 0 and
                    close[i] > weekly_ema_6h[i]):  # Aligned with weekly trend
                    position = 1
                    signals[i] = 0.25
                # Short: Bear Power negative AND falling (bears in control and gaining strength)
                #        Bull Power positive AND rising (bulls weak and weakening)
                elif (bear_power[i] < 0 and bear_power_slope[i] < 0 and 
                      bull_power[i] > 0 and bull_power_slope[i] > 0 and
                      close[i] < weekly_ema_6h[i]):  # Aligned with weekly trend
                    position = -1
                    signals[i] = -0.25
    
    return signals