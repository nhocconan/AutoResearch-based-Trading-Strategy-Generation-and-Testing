#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + VWAP + volume confirmation
# - Elder Ray: Bull Power = Close - EMA13(High), Bear Power = EMA13(Low) - Close
# - Long when Bull Power > 0 AND price > VWAP AND volume > 1.5x 20-period average
# - Short when Bear Power > 0 AND price < VWAP AND volume > 1.5x 20-period average
# - Exit when Elder Ray power reverses sign OR price crosses VWAP in opposite direction
# - VWAP acts as dynamic support/resistance, Elder Ray measures trend strength
# - Volume confirmation ensures breakouts have conviction
# - Targets 12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Works in both bull and bear markets by measuring institutional buying/selling pressure

name = "6h_elder_ray_vwap_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute 13-period EMA of high and low for Elder Ray
    ema13_high = pd.Series(prices['high']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_low = pd.Series(prices['low']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = prices['close'].values - ema13_high  # Buy power
    bear_power = ema13_low - prices['close'].values   # Sell power
    
    # Pre-compute VWAP (typical price * volume) cumulative
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    vwap_numerator = (typical_price * prices['volume']).cumsum()
    vwap_denominator = prices['volume'].cumsum()
    vwap = (vwap_numerator / vwap_denominator).values
    
    # Volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vwap[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power positive AND price above VWAP AND volume spike
            if (bull_power[i] > 0 and 
                prices['close'].iloc[i] > vwap[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power positive AND price below VWAP AND volume spike
            elif (bear_power[i] > 0 and 
                  prices['close'].iloc[i] < vwap[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Elder Ray power reverses (loss of momentum)
            # 2. Price crosses VWAP in opposite direction (trend change)
            if position == 1:  # Long position
                if (bull_power[i] <= 0 or 
                    prices['close'].iloc[i] < vwap[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (bear_power[i] <= 0 or 
                    prices['close'].iloc[i] > vwap[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals