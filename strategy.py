#!/usr/bin/env python3
"""
Experiment #8719: 6h Williams Alligator + Elder Ray + 12h trend filter + volume confirmation
Hypothesis: Williams Alligator identifies trend direction and convergence/divergence,
Elder Ray measures bull/bear power behind price action, and 12h EMA filter ensures
trading in direction of higher timeframe trend. Volume confirmation adds institutional
validation. Designed for 6h timeframe to balance signal quality and trade frequency.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while
maintaining statistical validity across bull/bear/sideways markets.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8719_6h_alligator_elder_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # Blue line: 13-period SMMA smoothed 8 bars ahead
ALLIGATOR_TEETH_PERIOD = 8  # Red line: 8-period SMMA smoothed 5 bars ahead
ALLIGATOR_LIPS_PERIOD = 5   # Green line: 5-period SMMA smoothed 3 bars ahead
ELDER_RAY_PERIOD = 13       # EMA period for Bull/Bear Power
TREND_EMA_PERIOD = 50       # 12h EMA for trend filter
VOLUME_MA_PERIOD = 20       # Volume moving average
VOLUME_THRESHOLD = 1.5      # Volume must be 1.5x MA to confirm
SIGNAL_SIZE = 0.25          # Position size (25% of capital)

def smma(data, period):
    """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    result = np.full(len(data), np.nan)
    # First value is simple SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Close) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def calculate_elder_ray(high, low, close, period):
    """Calculate Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power, ema

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=TREND_EMA_PERIOD, adjust=False, min_periods=TREND_EMA_PERIOD).mean().values
    
    # Price relative to 12h EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_12h > ema_12h, 1, 
                     np.where(close_12h < ema_12h, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_12h, price_vs_ema)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator lines (using SMMA)
    jaw = smma(close, ALLIGATOR_JAW_PERIOD)
    teeth = smma(close, ALLIGATOR_TEETH_PERIOD)
    lips = smma(close, ALLIGATOR_LIPS_PERIOD)
    
    # Shift the lines as per Williams Alligator specification
    # Jaw: 13-period SMMA smoothed 8 bars ahead
    # Teeth: 8-period SMMA smoothed 5 bars ahead  
    # Lips: 5-period SMMA smoothed 3 bars ahead
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Elder Ray
    bull_power, bear_power, ema_elder = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period - need enough data for all indicators
    start = max(
        ALLIGATOR_JAW_PERIOD + 8,   # Jaw needs shift
        ALLIGATOR_TEETH_PERIOD + 5, # Teeth needs shift
        ALLIGATOR_LIPS_PERIOD + 3,  # Lips needs shift
        ELDER_RAY_PERIOD,
        TREND_EMA_PERIOD,
        VOLUME_MA_PERIOD
    ) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Skip if any Alligator line is not ready (NaN due to shifting)
        if np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Williams Alligator conditions
        # Mouth open (trending): Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        # Mouth closed (convergence): lines intertwined
        lips_above_teeth = lips_shifted[i] > teeth_shifted[i]
        teeth_above_jaw = teeth_shifted[i] > jaw_shifted[i]
        lips_below_teeth = lips_shifted[i] < teeth_shifted[i]
        teeth_below_jaw = teeth_shifted[i] < jaw_shifted[i]
        
        # Strong bullish alignment: Lips > Teeth > Jaw (alligator eating up)
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        # Strong bearish alignment: Lips < Teeth < Jaw (alligator eating down)
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        # Elder Ray conditions
        # Bull power increasing and positive = bullish momentum
        # Bear power negative and decreasing (more negative) = bearish momentum
        bull_power_rising = i > 0 and bull_power[i] > bull_power[i-1]
        bear_power_falling = i > 0 and bear_power[i] < bear_power[i-1]  # More negative
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Determine market bias from 12h EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 12h price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 12h price below EMA50
        
        # Entry conditions
        # Long: Bullish Alligator alignment + rising bull power + bullish 12h bias + volume
        long_entry = (bullish_alignment and 
                     bull_power_rising and 
                     bull_bias and 
                     volume_confirmed)
        
        # Short: Bearish Alligator alignment + falling bear power + bearish 12h bias + volume
        short_entry = (bearish_alignment and 
                      bear_power_falling and 
                      bear_bias and 
                      volume_confirmed)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Stay in long position until opposite signal or convergence
            # Exit if Alligator lines converge (mouth closes) or bearish alignment
            lips_teeth_cross = lips_shifted[i] < teeth_shifted[i]  # Lips crossed below teeth
            teeth_jaw_cross = teeth_shifted[i] < jaw_shifted[i]    # Teeth crossed below jaw
            bearish_convergence = lips_teeth_cross and teeth_jaw_cross
            
            if bearish_convergence or bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Stay in short position until opposite signal or convergence
            # Exit if Alligator lines converge (mouth closes) or bullish alignment
            lips_teeth_cross = lips_shifted[i] > teeth_shifted[i]  # Lips crossed above teeth
            teeth_jaw_cross = teeth_shifted[i] > jaw_shifted[i]    # Teeth crossed above jaw
            bullish_convergence = lips_teeth_cross and teeth_jaw_cross
            
            if bullish_convergence or bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals