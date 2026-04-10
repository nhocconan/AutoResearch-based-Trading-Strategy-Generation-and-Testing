#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d trend filter + volume confirmation
# - Williams Alligator: Jaw (EMA13, 8-period shift), Teeth (EMA8, 5-period shift), Lips (EMA5, 3-period shift)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND 1d close > 1d EMA50 AND volume > 1.5x average
# - Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND 1d close < 1d EMA50 AND volume > 1.5x average
# - Exit when Alligator lines converge (Lips crosses Teeth) OR volume drops below 1.0x average
# - Uses 1d trend filter to avoid counter-trend trades in bear markets (2025+)
# - Volume confirmation reduces false signals and targets 12-25 trades/year (48-100 total over 4 years)
# - Tight entry conditions to avoid fee drag while maintaining edge in both bull and bear regimes

name = "12h_1d_alligator_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 1.0x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (1.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute Williams Alligator on 12h timeframe
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Jaw: EMA(13) shifted by 8 bars
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # Shift right by 8 bars (future becomes past)
    jaw[:8] = np.nan  # First 8 values invalid after shift
    
    # Teeth: EMA(8) shifted by 5 bars
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # Shift right by 5 bars
    teeth[:5] = np.nan  # First 5 values invalid after shift
    
    # Lips: EMA(5) shifted by 3 bars
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # Shift right by 3 bars
    lips[:3] = np.nan  # First 3 values invalid after shift
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long entry: bullish alignment AND price > Lips AND 1d uptrend AND volume spike
            if (bullish_alignment and 
                prices['close'].iloc[i] > lips[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish alignment AND price < Lips AND 1d downtrend AND volume spike
            elif (bearish_alignment and 
                  prices['close'].iloc[i] < lips[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # Stay flat
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Alligator lines converge (Lips crosses Teeth)
            # 2. Volume drops below 1.0x average (loss of momentum)
            lips_teeth_cross = (position == 1 and lips[i] < teeth[i]) or (position == -1 and lips[i] > teeth[i])
            
            if lips_teeth_cross or vol_weak.iloc[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25  # Hold position
    
    return signals