#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with Elder Ray confirmation and volume filter
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND volume > 1.5x 20-period avg
# - Short: Lips < Teeth < Jaw (bearish alignment) AND Bear Power < 0 AND volume > 1.5x 20-period avg
# - Exit: Alligator alignment breaks (Lips crosses Teeth) OR Elder Power reverses
# - Uses 1w EMA(34) trend filter: price > EMA for long bias, price < EMA for short bias
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 15-25 trades/year (60-100 total over 4 years) to stay within fee drag limits
# - Williams Alligator identifies trend absence/presence; Elder Ray measures bull/bear power; volume confirms conviction

name = "1d_1w_alligator_elder_volume_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return signals
    
    # Pre-compute 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Pre-compute Williams Alligator components (1d timeframe)
    # Median price = (High + Low) / 2
    median_price = (high + low) / 2
    
    # Alligator Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    jaw_values = jaw.values
    
    # Alligator Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    teeth_values = teeth.values
    
    # Alligator Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    lips_values = lips.values
    
    # Pre-compute Elder Ray components
    # EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA(13)
    bull_power = high - ema_13
    # Bear Power = Low - EMA(13)
    bear_power = low - ema_13
    
    # Pre-compute volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(lips_values[i]) or np.isnan(teeth_values[i]) or np.isnan(jaw_values[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Williams Alligator values
        lips_val = lips_values[i]
        teeth_val = teeth_values[i]
        jaw_val = jaw_values[i]
        
        # Elder Ray values
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1w EMA trend bias
        ema_bias_long = close_price > ema_34_1w_aligned[i]
        ema_bias_short = close_price < ema_34_1w_aligned[i]
        
        # Alligator alignment conditions
        alligator_bullish = lips_val > teeth_val > jaw_val  # Lips above Teeth above Jaw
        alligator_bearish = lips_val < teeth_val < jaw_val  # Lips below Teeth below Jaw
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bullish Alligator alignment + positive Bull Power + volume confirmation + long bias
        if alligator_bullish and bull_power_val > 0 and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short: Bearish Alligator alignment + negative Bear Power + volume confirmation + short bias
        if alligator_bearish and bear_power_val < 0 and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Alligator alignment breaks bearish OR Bull Power turns negative
            exit_long = not alligator_bullish or bull_power_val <= 0
        elif position == -1:
            # Exit short if Alligator alignment breaks bullish OR Bear Power turns positive
            exit_short = not alligator_bearish or bear_power_val >= 0
        
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