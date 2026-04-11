#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray combination with 12h volume confirmation
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long: Alligator aligned (Lips>Teeth>Jaw) + Bull Power > 0 + 12h volume > 1.5x 20-period average
# - Short: Alligator aligned inverse (Lips<Teeth<Jaw) + Bear Power < 0 + 12h volume > 1.5x 20-period average
# - Exit: Opposite Alligator alignment or Elder Ray power crosses zero
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - Alligator identifies trend direction and alignment, Elder Ray measures bull/bear power
# - Volume confirmation filters weak signals, 12h HTF reduces noise
# - Works in both bull (strong Alligator alignment) and bear (inverse alignment) markets

name = "4h_12h_alligator_elder_ray_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute Williams Alligator on 4h timeframe
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Pre-compute Elder Ray on 4h timeframe
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # Williams Alligator alignment
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        alligator_long = lips_above_teeth and teeth_above_jaw
        alligator_short = lips_below_teeth and teeth_below_jaw
        
        # Elder Ray power
        bull_positive = bull_power[i] > 0
        bear_negative = bear_power[i] < 0
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Alligator aligned up + Bull Power positive + volume confirmation
        if alligator_long and bull_positive and vol_confirm:
            enter_long = True
        
        # Short: Alligator aligned down + Bear Power negative + volume confirmation
        if alligator_short and bear_negative and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Alligator alignment breaks down or Bull Power turns negative
            exit_long = not (alligator_long and bull_positive)
        elif position == -1:
            # Exit short if Alligator alignment breaks up or Bear Power turns positive
            exit_short = not (alligator_short and bear_negative)
        
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