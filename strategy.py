#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray combo with 1d volume confirmation
# - Williams Alligator (Jaw=13, Teeth=8, Lips=5) defines trend direction and filters whipsaws
# - Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures trend strength
# - Long: Alligator bullish (Lips > Teeth > Jaw) + Bull Power > 0 + Bear Power rising + volume > 1.2x 20-period 1d average
# - Short: Alligator bearish (Lips < Teeth < Jaw) + Bear Power < 0 + Bull Power falling + volume > 1.2x 20-period 1d average
# - Exit: Opposite Alligator alignment or Elder Ray power crossover
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams Alligator excels in trending markets while filtering sideways chop
# - Elder Ray adds momentum confirmation to avoid false Alligator signals
# - 12h timeframe balances trade frequency with responsiveness to major trend changes

name = "12h_1d_alligator_elder_ray_volume_v1"
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
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Williams Alligator on 12h timeframe
    # Jaw (Blue): 13-period SMMA smoothed 8 periods ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean().values
    
    # Teeth (Red): 8-period SMMA smoothed 5 periods ahead
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean().values
    
    # Lips (Green): 5-period SMMA smoothed 3 periods ahead
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean().values
    
    # Pre-compute Elder Ray on 12h timeframe
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Smoothed Bull/Bear Power (3-period SMA for signal clarity)
    bull_power_smooth = pd.Series(bull_power).rolling(window=3, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).rolling(window=3, min_periods=3).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Williams Alligator conditions
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray conditions
        bull_power_positive = bull_power_smooth[i] > 0
        bear_power_negative = bear_power_smooth[i] < 0
        bull_power_rising = i > 100 and bull_power_smooth[i] > bull_power_smooth[i-1]
        bull_power_falling = i > 100 and bull_power_smooth[i] < bull_power_smooth[i-1]
        bear_power_falling = i > 100 and bear_power_smooth[i] < bear_power_smooth[i-1]
        bear_power_rising = i > 100 and bear_power_smooth[i] > bear_power_smooth[i-1]
        
        # Volume confirmation: current volume > 1.2x 20-period 1d average
        vol_confirm = volume_current > 1.2 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Alligator bullish + Bull Power positive + Bull Power rising + volume confirmation
        if alligator_bullish and bull_power_positive and bull_power_rising and vol_confirm:
            enter_long = True
        
        # Short: Alligator bearish + Bear Power negative + Bull Power falling + volume confirmation
        if alligator_bearish and bear_power_negative and bull_power_falling and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Alligator turns bearish OR Bull Power becomes negative
            exit_long = not alligator_bullish or bull_power_smooth[i] <= 0
        elif position == -1:
            # Exit short if Alligator turns bullish OR Bear Power becomes positive
            exit_short = not alligator_bearish or bear_power_smooth[i] >= 0
        
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