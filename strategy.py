#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# - Long when price breaks above 20-period Donchian high AND 1d EMA50 rising AND volume > 2.0x 20-bar avg
# - Short when price breaks below 20-period Donchian low AND 1d EMA50 falling AND volume > 2.0x 20-bar avg
# - Exit with ATR-based trailing stop (3*ATR) or opposite Donchian breakout
# - Uses 1d EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Donchian breakouts capture strong moves; trend filter improves win rate in bear markets

name = "12h_1d_donchian_breakout_volume_trend_v1"
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
    
    # Pre-compute Donchian channels (20-period) from 12h data
    high_12h = prices['high'].rolling(window=20, min_periods=20).max().values
    low_12h = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute ATR(14) for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian high AND 1d uptrend with volume spike
            if (prices['close'].iloc[i] > high_12h[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and  # price above 1d EMA50
                vol_spike.iloc[i]):
                position = 1
                entry_price = prices['close'].iloc[i]
                highest_since_entry = entry_price
                signals[i] = 0.25
            # Short when price breaks below Donchian low AND 1d downtrend with volume spike
            elif (prices['close'].iloc[i] < low_12h[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and  # price below 1d EMA50
                  vol_spike.iloc[i]):
                position = -1
                entry_price = prices['close'].iloc[i]
                lowest_since_entry = entry_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - manage trade
            # Update highest/lowest since entry
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
            else:
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
            
            # Check stoploss conditions
            stop_signal = False
            if position == 1:  # Long position
                # ATR trailing stop: exit if price drops 3*ATR from highest since entry
                if prices['close'].iloc[i] < highest_since_entry - 3.0 * atr[i]:
                    stop_signal = True
                # Opposite Donchian breakout: exit if price breaks below Donchian low
                elif prices['close'].iloc[i] < low_12h[i]:
                    stop_signal = True
            elif position == -1:  # Short position
                # ATR trailing stop: exit if price rises 3*ATR from lowest since entry
                if prices['close'].iloc[i] > lowest_since_entry + 3.0 * atr[i]:
                    stop_signal = True
                # Opposite Donchian breakout: exit if price breaks above Donchian high
                elif prices['close'].iloc[i] > high_12h[i]:
                    stop_signal = True
            
            if stop_signal:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals