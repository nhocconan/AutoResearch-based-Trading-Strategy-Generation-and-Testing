#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray combo with 1d EMA50 trend filter
# Long: Alligator bullish (jaw < teeth < lips) AND Elder Ray bullish (bull power > 0 AND bear power < 0) AND price > 1d EMA50
# Short: Alligator bearish (jaw > teeth > lips) AND Elder Ray bearish (bull power < 0 AND bear power > 0) AND price < 1d EMA50
# Exit: Alligator changes direction OR price crosses 1d EMA50
# Using 1d HTF for trend filter provides stability against whipsaws in choppy markets
# Williams Alligator identifies trend initiation and continuation
# Elder Ray measures bull/bear power behind the move
# Discrete position sizing: 0.25 for long/short, 0.0 for flat to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_WilliamsAlligator_ElderRay_1dEMA50_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs of median price
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    median_price = (high + low) / 2.0
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift the lines (jaw shifted 8, teeth shifted 5, lips shifted 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 13, 8, 5, 60)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any indicator is not ready
        if np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions: Alligator turns bearish OR price crosses below 1d EMA50
            alligator_bearish = jaw_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > lips_shifted[i]
            price_below_ema = curr_close < curr_ema_1d
            if alligator_bearish or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Alligator turns bullish OR price crosses above 1d EMA50
            alligator_bullish = jaw_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < lips_shifted[i]
            price_above_ema = curr_close > curr_ema_1d
            if alligator_bullish or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Alligator conditions
            alligator_bullish = jaw_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < lips_shifted[i]
            alligator_bearish = jaw_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > lips_shifted[i]
            
            # Elder Ray conditions
            elder_bullish = curr_bull_power > 0 and curr_bear_power < 0
            elder_bearish = curr_bull_power < 0 and curr_bear_power > 0
            
            # Long entry: Alligator bullish AND Elder Ray bullish AND price > 1d EMA50
            if alligator_bullish and elder_bullish and curr_close > curr_ema_1d:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Alligator bearish AND Elder Ray bearish AND price < 1d EMA50
            elif alligator_bearish and elder_bearish and curr_close < curr_ema_1d:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals