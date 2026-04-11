#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray combination with volume confirmation
# - Uses Alligator (Jaw/Teeth/Lips) to identify trend direction and avoid choppy markets
# - Elder Ray (Bull Power/Bear Power) to measure trend strength relative to EMA
# - Volume confirmation to filter weak signals
# - Primary timeframe: 12h (lower frequency = less fee drag, better generalization)
# - HTF: 1d for trend context
# - Designed to work in both bull (trend following) and bear (mean reversion in extremes) markets
# - Discrete position sizing (±0.25) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "12h_alligator_elder_ray_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1d data ONCE before loop for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA for Elder Ray calculation
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 1d volume average for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Williams Alligator on 12h timeframe
    # Jaw (blue): 13-period SMMA smoothed by 8 periods
    # Teeth (red): 8-period SMMA smoothed by 5 periods  
    # Lips (green): 5-period SMMA smoothed by 3 periods
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CURRENT) / N
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate Alligator components
    jaw = smma(close, 13)  # 13-period SMMA
    jaw = smma(jaw, 8)     # smoothed by 8
    
    teeth = smma(close, 8)   # 8-period SMMA
    teeth = smma(teeth, 5)   # smoothed by 5
    
    lips = smma(close, 5)    # 5-period SMMA
    lips = smma(lips, 3)     # smoothed by 3
    
    # Pre-compute Elder Ray components on 12h
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_12 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_12
    bear_power = low - ema_12
    
    # Smooth the power indicators
    bull_power_smooth = pd.Series(bull_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_13_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or
            np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.2x 20-period average
        vol_confirm = volume_current > 1.2 * volume_sma_20_aligned[i]
        
        # Alligator trend detection
        # When Lips > Teeth > Jaw = uptrend (green above red above blue)
        # When Lips < Teeth < Jaw = downtrend (green below red below blue)
        # When intertwined = choppy/no trend
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        uptrend = lips_above_teeth and teeth_above_jaw
        downtrend = lips_below_teeth and teeth_below_jaw
        
        # Elder Ray trend strength
        strong_bull = bull_power_smooth[i] > 0 and bull_power_smooth[i] > bear_power_smooth[i]
        strong_bear = bear_power_smooth[i] < 0 and abs(bear_power_smooth[i]) > abs(bull_power_smooth[i])
        
        # HTF trend filter from 1d EMA
        htf_uptrend = close_price > ema_13_aligned[i]
        htf_downtrend = close_price < ema_13_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Alligator uptrend + Elder Ray bullish + HTF uptrend + volume confirmation
        if uptrend and strong_bull and htf_uptrend and vol_confirm:
            enter_long = True
        
        # Short: Alligator downtrend + Elder Ray bearish + HTF downtrend + volume confirmation
        if downtrend and strong_bear and htf_downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: reverse signals or loss of momentum
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if trend weakens or reverses
            exit_long = (not uptrend) or (not strong_bull) or (not htf_uptrend) or downtrend
        elif position == -1:
            # Exit short if trend weakens or reverses
            exit_short = (not downtrend) or (not strong_bear) or (not htf_downtrend) or uptrend
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals