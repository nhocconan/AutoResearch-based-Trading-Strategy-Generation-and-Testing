#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator with daily volume confirmation and weekly trend filter
# Long when price is above Alligator teeth (red line) with volume expansion and weekly bullish trend
# Short when price is below Alligator teeth with volume expansion and weekly bearish trend
# Exit when price crosses Alligator jaws (blue line)
# Uses Williams Alligator (13,8,5 SMAs) to identify trend phases and avoid choppy markets
# Target: 20-50 trades per symbol over 4 years (5-12.5/year) to minimize fee drag
# Williams Alligator is effective in both trending and ranging markets when combined with volume and trend filters

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and weekly data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate Williams Alligator on 4h timeframe
    # Jaw (Blue Line): 13-period SMMA, smoothed 8 periods ahead
    # Teeth (Red Line): 8-period SMMA, smoothed 5 periods ahead
    # Lips (Green Line): 5-period SMMA, smoothed 3 periods ahead
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate median price (typical price) for Alligator
    typical_price_4h = (high_4h + low_4h + close_4h) / 3
    
    # Calculate SMMA (Smoothed Moving Average) - equivalent to Wilder's smoothing
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values are smoothed
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate Alligator lines
    jaw = smma(typical_price_4h, 13)  # Blue line
    teeth = smma(typical_price_4h, 8)  # Red line
    lips = smma(typical_price_4h, 5)   # Green line
    
    # Shift jaw forward by 8, teeth by 5, lips by 3 (as per Williams Alligator)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # For the shifted periods, we need to handle the NaN propagation
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly EMA for trend filter (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips_shifted)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max shift + period)
    start = 30  # for Alligator calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]  # Current 4h volume
        
        if position == 0:
            # Long setup: price above teeth (red line) with volume expansion and weekly bullish trend
            # Alligator lines should be aligned: lips > teeth > jaw (bullish alignment)
            if (price > teeth_aligned[i] and 
                lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and  # Bullish alignment
                vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and        # Volume expansion
                price > ema_weekly_aligned[i]):                        # Price above weekly EMA for bullish trend
                position = 1
                signals[i] = position_size
            # Short setup: price below teeth (red line) with volume expansion and weekly bearish trend
            # Alligator lines should be aligned: lips < teeth < jaw (bearish alignment)
            elif (price < teeth_aligned[i] and 
                  lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and  # Bearish alignment
                  vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and        # Volume expansion
                  price < ema_weekly_aligned[i]):                        # Price below weekly EMA for bearish trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lips (green line) or Alligator lines lose bullish alignment
            if (price < lips_aligned[i] or 
                not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])):  # Lost bullish alignment
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above lips (green line) or Alligator lines lose bearish alignment
            if (price > lips_aligned[i] or 
                not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])):  # Lost bearish alignment
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsAlligator_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0