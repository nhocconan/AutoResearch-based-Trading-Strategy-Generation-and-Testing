#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout with volume confirmation and ATR trailing stop
# - Uses 1w HTF for Donchian(20) upper/lower channels (based on completed weekly candles)
# - Long when price breaks above Donchian upper channel with volume > 2.0x 20-period average
# - Short when price breaks below Donchian lower channel with volume > 2.0x 20-period average
# - ATR(14) trailing stop: exit long at 3.0x ATR below highest high since entry, exit short at 3.0x ATR above lowest low since entry
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: Donchian channels adapt to volatility, volume confirmation filters false breakouts
# - Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years)

name = "1d_1w_donchian_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian channel (20-period) - based on completed weekly candles
    # Upper channel = highest high of last 20 weeks
    # Lower channel = lowest low of last 20 weeks
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (wait for completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_max_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_min_20)
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or
            vol_ma_20[i] <= 0 or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 3.0x ATR from highest high
            if close[i] < highest_high_since_entry - 3.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 3.0x ATR from lowest low
            if close[i] > lowest_low_since_entry + 3.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout with volume confirmation
            if volume_confirmed:
                # Long entry: price breaks above Donchian upper channel
                if close[i] > donchian_high_aligned[i]:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian lower channel
                elif close[i] < donchian_low_aligned[i]:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals