#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Entry: Price breaks above/below 20-day Donchian channel with volume > 1.5x 20-day average
# - Filter: Only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - Exit: ATR(14) trailing stop (2.5x) or opposite Donchian breakout
# - Position sizing: 0.25 discrete to minimize fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) within HARD MAX: 150 total
# - Works in bull/bear: Donchian captures breakouts, weekly EMA filter avoids counter-trend trades

name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    # ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current values
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        vol_spike = volume_spike[i]
        
        # Weekly trend filter
        weekly_uptrend = close_price > ema_50_aligned[i]
        weekly_downtrend = close_price < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high with volume spike and weekly uptrend
            if high_price > donchian_high[i] and vol_spike and weekly_uptrend:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else close_price
                highest_since_entry = high_price
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume spike and weekly downtrend
            elif low_price < donchian_low[i] and vol_spike and weekly_downtrend:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else close_price
                lowest_since_entry = low_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update highest/lowest since entry
            if position == 1:
                highest_since_entry = max(highest_since_entry, high_price)
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = close_price < highest_since_entry - 2.5 * atr[i]
                # Opposite Donchian breakout exit
                opposite_breakout = low_price < donchian_low[i]
                exit_condition = trailing_stop or opposite_breakout
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, low_price)
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = close_price > lowest_since_entry + 2.5 * atr[i]
                # Opposite Donchian breakout exit
                opposite_breakout = high_price > donchian_high[i]
                exit_condition = trailing_stop or opposite_breakout
            
            if exit_condition:
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