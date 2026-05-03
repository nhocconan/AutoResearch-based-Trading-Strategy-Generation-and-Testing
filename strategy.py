#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# In bull regime (price > 1d EMA50), go long when Alligator lines are aligned bullish (jaw < teeth < lips) with volume spike.
# In bear regime (price < 1d EMA50), go short when Alligator lines are aligned bearish (jaw > teeth > lips) with volume spike.
# Uses 1d EMA50 for regime filter, 12h Williams Alligator for entry signals, and 12h volume spike for confirmation.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Focus on BTC/ETH as primary symbols.

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "12h"
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
    
    # Calculate 1d EMA50 for regime filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Get 12h data for Williams Alligator (jaw=13, teeth=8, lips=5 SMAs of median price)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate median price for Alligator
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # Williams Alligator lines: jaw (13), teeth (8), lips (5) SMAs of median price
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 12h (wait for 12h bar to complete)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
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
        
        # Alligator alignment conditions
        bullish_alignment = (jaw_val < teeth_val) and (teeth_val < lips_val)
        bearish_alignment = (jaw_val > teeth_val) and (teeth_val > lips_val)
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: bullish Alligator alignment with volume spike
            long_entry = bullish_alignment and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: bearish Alligator alignment with volume spike
            short_entry = bearish_alignment and vol_spike
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
            # Exit on bearish Alligator alignment (failure of bullish structure) or regime change to bear
            if bearish_alignment or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on bullish Alligator alignment (failure of bearish structure) or regime change to bull
            if bullish_alignment or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals