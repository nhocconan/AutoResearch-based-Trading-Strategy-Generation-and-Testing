#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume confirmation
# - Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) in 12h uptrend (close > EMA50) with volume > 1.5x 20-bar avg
# - Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) in 12h downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - 12h trend filter reduces false signals in counter-trend markets
# - Elder Ray measures power of bulls/bears relative to EMA, effective in 6h timeframe
# - Volume confirmation ensures institutional participation

name = "6h_12h_elderray_power_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h EMA(13) for Elder Ray calculation
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_13_12h)
    
    # 12h Elder Ray components
    bull_power_12h = high_12h - ema_13_12h  # High - EMA13
    bear_power_12h = ema_13_12h - low_12h   # EMA13 - Low
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # 12h volume confirmation: > 1.5x 20-period average
    avg_volume_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.5 * avg_volume_20_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_13_12h_aligned[i]) or 
            np.isnan(bull_power_12h_aligned[i]) or np.isnan(bear_power_12h_aligned[i]) or 
            np.isnan(vol_spike_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Bull Power > 0 and Bear Power < 0 (bullish) in 12h uptrend with volume spike
            if (bull_power_12h_aligned[i] > 0 and 
                bear_power_12h_aligned[i] < 0 and 
                prices['close'].iloc[i] > ema_50_12h_aligned[i] and 
                vol_spike_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Bear Power > 0 and Bull Power < 0 (bearish) in 12h downtrend with volume spike
            elif (bear_power_12h_aligned[i] > 0 and 
                  bull_power_12h_aligned[i] < 0 and 
                  prices['close'].iloc[i] < ema_50_12h_aligned[i] and 
                  vol_spike_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when power divergence disappears (loss of momentum)
            if position == 1 and (bull_power_12h_aligned[i] <= 0 or bear_power_12h_aligned[i] >= 0):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (bear_power_12h_aligned[i] <= 0 or bull_power_12h_aligned[i] >= 0):
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals