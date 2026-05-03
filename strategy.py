#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# In bull regime (price > 1d EMA50), go long when Jaw < Teeth < Lips (Alligator awake) with volume spike.
# In bear regime (price < 1d EMA50), go short when Jaw > Teeth > Lips (Alligator awake) with volume spike.
# Uses Williams Alligator (SMAs with 5,8,13 periods) from 4h data, 1d EMA50 for regime filter,
# and 4h volume spike for confirmation. Designed for 75-200 total trades over 4 years.
# Focus on BTC/ETH as primary symbols, avoids SOL-only bias.

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams Alligator on 4h data
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA (Smoothed Moving Average) approximation using EMA with alpha=1/period
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Calculate volume regime: current 4h volume > 1.8x 30-period MA
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get current values
        close_val = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA50, bear if close < 1d EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Alligator awake conditions: Jaw, Teeth, Lips are aligned and trending
        bull_alligator = (jaw_val < teeth_val) and (teeth_val < lips_val)
        bear_alligator = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: bull regime + bullish Alligator alignment + volume spike
            long_entry = bull_alligator and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: bear regime + bearish Alligator alignment + volume spike
            short_entry = bear_alligator and vol_spike
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on bearish Alligator alignment or regime change to bear
            if bear_alligator or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on bullish Alligator alignment or regime change to bull
            if bull_alligator or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals