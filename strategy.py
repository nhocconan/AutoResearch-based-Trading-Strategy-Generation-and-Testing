#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + ATR Trailing Stop
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1d volume > 2.0x 20-bar avg
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1d volume > 2.0x 20-bar avg
# - Exit via ATR trailing stop: 3*ATR from extreme price
# - Uses 12h timeframe for lower trade frequency (~20-40/year) to minimize fee drag
# - Alligator catches trends; volume confirms conviction; ATR stop manages risk
# - Works in both bull (trend following) and bear (short trends) markets

name = "12h_1d_alligator_volume_atrstop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams Alligator on 12h timeframe
    # Median price = (high + low) / 2
    median_price = (prices['high'] + prices['low']) / 2
    
    # Jaw: 13-period SMA, smoothed by 8 periods
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = pd.Series(jaw).rolling(window=8, min_periods=8).mean()
    
    # Teeth: 8-period SMA, smoothed by 5 periods
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = pd.Series(teeth).rolling(window=5, min_periods=5).mean()
    
    # Lips: 5-period SMA, smoothed by 3 periods
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = pd.Series(lips).rolling(window=3, min_periods=3).mean()
    
    # Aligator alignment conditions
    lips_above_teeth = lips > teeth
    teeth_above_jaw = teeth > jaw
    lips_below_teeth = lips < teeth
    teeth_below_jaw = teeth < jaw
    
    bullish_align = lips_above_teeth & teeth_above_jaw
    bearish_align = lips_below_teeth & teeth_below_jaw
    
    # Pre-compute 1d volume confirmation: > 2.0x 20-period average
    volume_20_avg = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'] > (2.0 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.values)
    
    # Pre-compute ATR(14) for trailing stop
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift())
    low_close = np.abs(prices['low'] - prices['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bullish_align.iloc[i]) or np.isnan(bearish_align.iloc[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(atr[i])):
            # Hold current position or flat
            signals[i] = 0.25 * position
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when bullish Alligator alignment AND volume spike
            if bullish_align.iloc[i] and vol_spike_1d_aligned[i]:
                position = 1
                entry_price = prices['close'].iloc[i]
                highest_since_entry = entry_price
                lowest_since_entry = entry_price
                signals[i] = 0.25
            # Short when bearish Alligator alignment AND volume spike
            elif bearish_align.iloc[i] and vol_spike_1d_aligned[i]:
                position = -1
                entry_price = prices['close'].iloc[i]
                highest_since_entry = entry_price
                lowest_since_entry = entry_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - update extremes and check trailing stop
            # Update highest/lowest since entry
            if position == 1:  # Long position
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                
                # Check trailing stop: 3*ATR below highest
                if prices['close'].iloc[i] < highest_since_entry - 3.0 * atr[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Short position
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                
                # Check trailing stop: 3*ATR above lowest
                if prices['close'].iloc[i] > lowest_since_entry + 3.0 * atr[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals