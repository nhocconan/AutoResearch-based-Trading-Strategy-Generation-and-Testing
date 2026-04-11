#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1w Elder Ray + volume confirmation
# - Williams Alligator (13,8,5 SMAs) from 12h: identifies trend direction and strength
# - Elder Ray (Bull/Bear Power) from 1w: measures bull/bear strength relative to weekly EMA13
# - Long when: Alligator bullish (jaw<teeth<lips) AND weekly Bull Power > 0 AND volume > 1.5x 20-period average
# - Short when: Alligator bearish (jaw>teeth>lips) AND weekly Bear Power < 0 AND volume > 1.5x 20-period average
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h
# - Volume confirmation ensures we only trade high-conviction signals
# - Works in bull markets (Alligator bullish + strong Bull Power) and bear markets (Alligator bearish + strong Bear Power)

name = "12h_1w_williams_alligator_elder_ray_v1"
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
    
    # Load 12h and 1w data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1w = get_htf_data(prices, '1w')
    if len(df_12h) < 50 or len(df_1w) < 50:
        return signals
    
    # Pre-compute 12h Williams Alligator (SMAs: 13,8,5)
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Jaw (13-period SMA), Teeth (8-period SMA), Lips (5-period SMA)
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    # Align 12h Alligator to 12h timeframe (no additional delay needed for SMAs)
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Pre-compute 1w Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # EMA13 for Elder Ray
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1w = high_1w - ema13_1w
    bear_power_1w = low_1w - ema13_1w
    
    # Align 1w Elder Ray to 12h timeframe (no additional delay needed)
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power_1w)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power_1w)
    
    # Pre-compute 12h volume SMA (20-period)
    volume_12h = df_12h['volume'].values
    volume_series = pd.Series(volume_12h)
    volume_sma_20_12h = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 12h volume SMA to 12h timeframe
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or np.isnan(lips_12h_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # Williams Alligator conditions
        alligator_bullish = (jaw_12h_aligned[i] < teeth_12h_aligned[i]) and (teeth_12h_aligned[i] < lips_12h_aligned[i])
        alligator_bearish = (jaw_12h_aligned[i] > teeth_12h_aligned[i]) and (teeth_12h_aligned[i] > lips_12h_aligned[i])
        
        # Elder Ray conditions
        bull_power_positive = bull_power_aligned[i] > 0
        bear_power_negative = bear_power_aligned[i] < 0
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Alligator bullish + Bull Power positive + volume confirmation
        if alligator_bullish and bull_power_positive and vol_confirm:
            enter_long = True
        
        # Short: Alligator bearish + Bear Power negative + volume confirmation
        if alligator_bearish and bear_power_negative and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Alligator alignment or volume collapse
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Alligator turns bearish OR volume collapses
            exit_long = alligator_bearish or (not vol_confirm)
        elif position == -1:
            # Exit short if Alligator turns bullish OR volume collapses
            exit_short = alligator_bullish or (not vol_confirm)
        
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