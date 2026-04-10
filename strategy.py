#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# - Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
# - Long when Lips > Teeth > Jaw AND price > Lips AND 1w EMA50 rising AND volume > 1.5x 20-bar avg
# - Short when Lips < Teeth < Jaw AND price < Lips AND 1w EMA50 falling AND volume > 1.5x 20-bar avg
# - Exit when Alligator lines re-cross (Lips crosses Teeth or Teeth crosses Jaw)
# - Uses 1w EMA50 for higher timeframe trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years)
# - Williams Alligator excels in trending markets; 1w filter ensures we trade with dominant trend

name = "12h_1w_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute Williams Alligator components (using SMMA - smoothed moving average)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # SMMA calculation (Smoothed Moving Average)
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Close) / Period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma(close, 13)
    jaw = np.roll(jaw, 8)  # Shifted 8 bars forward
    teeth = smma(close, 8)
    teeth = np.roll(teeth, 5)  # Shifted 5 bars forward
    lips = smma(close, 5)
    lips = np.roll(lips, 3)  # Shifted 3 bars forward
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Check for Alligator re-cross (exit condition)
        lips_teeth_cross = False
        teeth_jaw_cross = False
        
        if i > 0:
            # Lips crossing Teeth
            if ((lips[i-1] <= teeth[i-1] and lips[i] > teeth[i]) or
                (lips[i-1] >= teeth[i-1] and lips[i] < teeth[i])):
                lips_teeth_cross = True
            # Teeth crossing Jaw
            if ((teeth[i-1] <= jaw[i-1] and teeth[i] > jaw[i]) or
                (teeth[i-1] >= jaw[i-1] and teeth[i] < jaw[i])):
                teeth_jaw_cross = True
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when Lips > Teeth > Jaw (Alligator bullish alignment) AND price > Lips AND 1w uptrend with volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and  # Bullish alignment
                prices['close'].iloc[i] > lips[i] and  # Price above Lips
                prices['close'].iloc[i] > ema50_1w_aligned[i] and  # Price above 1w EMA50 (uptrend)
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when Lips < Teeth < Jaw (Alligator bearish alignment) AND price < Lips AND 1w downtrend with volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and  # Bearish alignment
                  prices['close'].iloc[i] < lips[i] and  # Price below Lips
                  prices['close'].iloc[i] < ema50_1w_aligned[i] and  # Price below 1w EMA50 (downtrend)
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on Alligator re-cross
            # Exit when Alligator lines re-cross (Lips-Teeth or Teeth-Jaw)
            exit_signal = lips_teeth_cross or teeth_jaw_cross
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals