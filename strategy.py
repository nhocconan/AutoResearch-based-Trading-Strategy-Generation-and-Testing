#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR stoploss
# - Long when price breaks above 20-period Donchian high with volume > 1.5x average
# - Short when price breaks below 20-period Donchian low with volume > 1.5x average
# - Exit when price retraces to midpoint of Donchian channel or volume drops
# - ATR-based trailing stop to manage risk (signal -> 0 when stop hit)
# - Targets 20-30 trades/year (80-120 total over 4 years) to avoid fee drag
# - Donchian breakouts work in both trending and ranging markets with volume filter

name = "4h_donchian_breakout_volume_atr_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute Donchian channels (20-period)
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Pre-compute ATR(14) for stoploss and position sizing
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift())
    low_close = np.abs(prices['low'] - prices['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(atr[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > Donchian high with volume spike AND 1d uptrend
            if (prices['close'].iloc[i] > high_20[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                entry_price = prices['close'].iloc[i]
                highest_since_entry = entry_price
                signals[i] = 0.25
            # Short breakdown: price < Donchian low with volume spike AND 1d downtrend
            elif (prices['close'].iloc[i] < low_20[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                entry_price = prices['close'].iloc[i]
                lowest_since_entry = entry_price
                signals[i] = -0.25
        else:  # Have position - manage trade
            if position == 1:  # Long position
                # Update highest price since entry
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                
                # Exit conditions:
                # 1. Price retraces to Donchian midpoint (mean reversion)
                # 2. Volume drops below average (loss of momentum)
                # 3. ATR trailing stop (2 * ATR below highest since entry)
                if (prices['close'].iloc[i] < donchian_mid[i] or 
                    vol_normal.iloc[i] or
                    prices['close'].iloc[i] < highest_since_entry - 2.0 * atr[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                # Update lowest price since entry
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                
                # Exit conditions:
                # 1. Price retraces to Donchian midpoint (mean reversion)
                # 2. Volume drops below average (loss of momentum)
                # 3. ATR trailing stop (2 * ATR above lowest since entry)
                if (prices['close'].iloc[i] > donchian_mid[i] or 
                    vol_normal.iloc[i] or
                    prices['close'].iloc[i] > lowest_since_entry + 2.0 * atr[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals