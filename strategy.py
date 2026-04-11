#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with Elder Ray filter and volume confirmation
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND volume > 1.5x 20-bar avg
# - Short: Lips < Teeth < Jaw (bearish alignment) AND Bear Power < 0 AND volume > 1.5x 20-bar avg
# - Exit: Alligator lines cross (Lips-Teeth or Teeth-Jaw crossover)
# - Uses 1d trend filter: price > EMA(50) for long bias, price < EMA(50) for short bias
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Alligator identifies trends, Elder Ray confirms strength, volume adds conviction
# - Works in both bull/bear by capturing strong trending moves with filter

name = "12h_1d_alligator_elder_volume_v1"
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
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute median price for Alligator (12h)
    median_price = (high + low) / 2
    
    # Williams Alligator lines (12h)
    # Jaw: 13-period SMMA, 8 periods ahead
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars ahead
    # Teeth: 8-period SMMA, 5 periods ahead
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars ahead
    # Lips: 5-period SMMA, 3 periods ahead
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars ahead
    
    # Pre-compute Elder Ray (12h)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Pre-compute volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Alligator alignment
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        # Elder Ray confirmation
        bull_confirm = bull_power[i] > 0
        bear_confirm = bear_power[i] < 0
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: bullish Alligator + Bull Power > 0 + volume + long bias
        if bullish_alignment and bull_confirm and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short: bearish Alligator + Bear Power < 0 + volume + short bias
        if bearish_alignment and bear_confirm and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions: Alligator lines cross (Lips-Teeth or Teeth-Jaw)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on bearish cross: Lips < Teeth OR Teeth < Jaw
            exit_long = (lips[i] < teeth[i]) or (teeth[i] < jaw[i])
        elif position == -1:
            # Exit short on bullish cross: Lips > Teeth OR Teeth > Jaw
            exit_short = (lips[i] > teeth[i]) or (teeth[i] > jaw[i])
        
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