#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + Volume Spike regime filter
# - Uses 1d timeframe for lower trade frequency and better generalization
# - Alligator (jaw/teeth/lips) defines trend direction and strength
# - Elder Ray (bull/bear power) confirms momentum with EMA(13)
# - Volume spike (>2x 20-period average) confirms institutional participation
# - Only trade when Alligator is aligned (trending) and Elder Ray confirms
# - Weekly HTF (1w) EMA(34) filter for higher timeframe trend alignment
# - Designed to work in both bull (strong Alligator alignment) and bear (clear Elder Ray divergence) markets
# - Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

name = "1d_alligator_elder_volume_v1"
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
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return signals
    
    # Pre-compute weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Pre-compute 1d Alligator (Smoothed Moving Average - SMA with specific periods)
    # Jaw: SMA(13, 8) - 13-period SMA shifted 8 bars forward
    # Teeth: SMA(8, 5) - 8-period SMA shifted 5 bars forward  
    # Lips: SMA(5, 3) - 5-period SMA shifted 3 bars forward
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Apply shifts (Alligator definition)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Fill NaN from rolling with forward fill equivalent
    for i in range(len(jaw)):
        if np.isnan(jaw[i]):
            jaw[i] = jaw_raw[i] if not np.isnan(jaw_raw[i]) else close[i]
        if np.isnan(teeth[i]):
            teeth[i] = teeth_raw[i] if not np.isnan(teeth_raw[i]) else close[i]
        if np.isnan(lips[i]):
            lips[i] = lips_raw[i] if not np.isnan(lips_raw[i]) else close[i]
    
    # Pre-compute 1d Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray confirmation: Bull Power > 0 and rising, Bear Power < 0 and falling
        # Simplified: Bull Power > 0 for long, Bear Power < 0 for short
        elder_long = bull_power[i] > 0
        elder_short = bear_power[i] < 0
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Weekly HTF trend filter: price above/below weekly EMA(34)
        weekly_long_bias = close_price > ema_34_1w_aligned[i]
        weekly_short_bias = close_price < ema_34_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Alligator aligned up + Elder Ray bullish + Volume spike + Weekly bias long
        if alligator_long and elder_long and vol_confirm and weekly_long_bias:
            enter_long = True
        
        # Short: Alligator aligned down + Elder Ray bearish + Volume spike + Weekly bias short
        if alligator_short and elder_short and vol_confirm and weekly_short_bias:
            enter_short = True
        
        # Exit conditions: Reverse Alligator signal or Elder Ray divergence
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Alligator loses alignment or Elder Ray turns bearish
            exit_long = not (alligator_long and elder_long)
        elif position == -1:
            # Exit short if Alligator loses alignment or Elder Ray turns bullish
            exit_short = not (alligator_short and elder_short)
        
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