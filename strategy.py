#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1-week trend filter and volume confirmation
# Williams Alligator consists of three SMAs: Jaw (13-period, 8-shift), Teeth (8-period, 5-shift), Lips (5-period, 3-shift)
# When all three lines are aligned and pointing up (Lips > Teeth > Jaw with upward slope), bullish trend
# When all three lines are aligned and pointing down (Lips < Teeth < Jaw with downward slope), bearish trend
# Weekly trend filter avoids counter-trend trades: only trade long when weekly trend up, short when weekly trend down
# Volume spike confirms institutional participation
# Target: 20-50 total trades over 4 years (5-12/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams Alligator components
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Jaw: 13-period SMA, shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift forward 8 bars
    jaw_values = jaw.values
    
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift forward 5 bars
    teeth_values = teeth.values
    
    # Lips: 5-period SMA, shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift forward 3 bars
    lips_values = lips.values
    
    # Calculate 1d ATR for stop sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: 1d volume > 2.0 x 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if (np.isnan(lips_values[i]) or np.isnan(teeth_values[i]) or np.isnan(jaw_values[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price level
        price = close[i]
        
        # Williams Alligator alignment and slope
        lips_above_teeth = lips_values[i] > teeth_values[i]
        teeth_above_jaw = teeth_values[i] > jaw_values[i]
        lips_below_teeth = lips_values[i] < teeth_values[i]
        teeth_below_jaw = teeth_values[i] < jaw_values[i]
        
        # Slope calculation (current vs previous)
        lips_rising = lips_values[i] > lips_values[i-1] if i > 0 else False
        lips_falling = lips_values[i] < lips_values[i-1] if i > 0 else False
        
        # Weekly trend: rising EMA50 = bullish, falling EMA50 = bearish
        weekly_trend_up = ema50_1w_aligned[i] > ema50_1w_aligned[i-1] if i > 0 else False
        weekly_trend_down = ema50_1w_aligned[i] < ema50_1w_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Bullish Alligator: Lips > Teeth > Jaw with upward slope
            bullish_alligator = lips_above_teeth and teeth_above_jaw and lips_rising
            # Bearish Alligator: Lips < Teeth < Jaw with downward slope
            bearish_alligator = lips_below_teeth and teeth_below_jaw and lips_falling
            
            # Long: bullish alligator, weekly trend up, volume spike
            if bullish_alligator and weekly_trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: bearish alligator, weekly trend down, volume spike
            elif bearish_alligator and weekly_trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2.5x ATR below entry) or bearish alligator signal
            if price <= entry_price - 2.5 * atr[i] or (lips_below_teeth and teeth_below_jaw):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2.5x ATR above entry) or bullish alligator signal
            if price >= entry_price + 2.5 * atr[i] or (lips_above_teeth and teeth_above_jaw):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_WeeklyEMA50Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0