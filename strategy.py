#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# - Uses 1w Camarilla pivot levels (H4/L4) to determine primary trend direction
# - Only take longs when price > weekly H4, shorts when price < weekly L4
# - 6h Donchian(20) breakout for entry timing: long on break above 20-period high, short on break below 20-period low
# - Volume confirmation: 6h volume > 1.8x 30-period average to ensure breakout strength
# - ATR(20) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: 50-150 total trades over 4 years (12-37/year) per 6h strategy guidelines
# - Novelty: Weekly Camarilla H4/L4 as trend filter prevents counter-trend trades in 6h Donchian breakouts
# - Works in bull markets: weekly H4 acts as dynamic resistance/support, breakouts continue trend
# - Works in bear markets: weekly L4 prevents false breakdowns, only takes shorts below significant weekly support

name = "6h_1w_donchian_camarilla_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels (H4, L4)
    # Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    camarilla_h4 = close_1w + (1.1 * (high_1w - low_1w) / 2)
    camarilla_l4 = close_1w - (1.1 * (high_1w - low_1w) / 2)
    
    # Align Camarilla levels to 6h timeframe (completed 1w bar only)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume > 1.8x 30-period average (volume confirmation)
    avg_volume_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * avg_volume_30)
    
    # 6h ATR(20) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high OR weekly L4 touch (mean reversion)
            if low[i] <= highest_since_entry - (2.0 * atr[i]) or \
               low[i] <= camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low OR weekly H4 touch (mean reversion)
            if high[i] >= lowest_since_entry + (2.0 * atr[i]) or \
               high[i] >= camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and weekly pivot filter
            # Long: price breaks above 20-period high AND volume spike AND price > weekly H4
            if high[i] >= highest_20[i] and volume_spike[i] and close[i] > camarilla_h4_aligned[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: price breaks below 20-period low AND volume spike AND price < weekly L4
            elif low[i] <= lowest_20[i] and volume_spike[i] and close[i] < camarilla_l4_aligned[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals