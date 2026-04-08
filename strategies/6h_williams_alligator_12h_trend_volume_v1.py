#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams Alligator + 12h Trend + Volume Confirmation
# Hypothesis: Williams Alligator identifies trends on 6h timeframe. We trade in direction of
# 12-hour EMA(50) when Alligator aligns (teeth > lips for long, teeth < lips for short),
# with volume confirmation. This avoids whipsaw in ranging markets. 6h timeframe balances
# responsiveness and noise reduction, suitable for both bull and bear markets.
# Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag.
name = "6h_williams_alligator_12h_trend_volume_v1"
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
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Williams Alligator on 6h timeframe
    # Jaw: SMA(13, 8) - median price smoothed
    # Teeth: SMA(8, 5) - median price smoothed
    # Lips: SMA(5, 3) - median price smoothed
    median_price = (high + low) / 2
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift forward 8 bars
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift forward 5 bars
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # shift forward 3 bars
    
    # 12-hour EMA(50) for trend filter
    daily_close = df_12h['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_6h = align_htf_to_ltf(prices, df_12h, daily_ema)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(daily_ema_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator reverses (teeth < lips) or trend turns bearish
            if teeth[i] < lips[i] or close[i] < daily_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Alligator reverses (teeth > lips) or trend turns bullish
            if teeth[i] > lips[i] or close[i] > daily_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation and Alligator alignment
            if vol_filter[i]:
                # Long: teeth > lips (bullish alignment) + price above 12H EMA
                if teeth[i] > lips[i] and close[i] > daily_ema_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: teeth < lips (bearish alignment) + price below 12H EMA
                elif teeth[i] < lips[i] and close[i] < daily_ema_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals