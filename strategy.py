#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w trend filter
# - Entry: Price breaks above/below 20-period Donchian channel on 12h timeframe
# - Confirmation: 1d volume > 2.0x 20-period average (strong participation)
# - Trend filter: Only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - Exit: ATR(14) trailing stop (2.5x) on 12h timeframe or opposite Donchian break
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) within HARD MAX: 200 total
# - Works in bull markets via breakouts, bear markets via short breakdowns with trend filter

name = "12h_1d_1w_donchian_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # Donchian high: rolling max of high
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume and its 20-period moving average for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 12h timeframe
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 12h ATR for trailing stop
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr_12h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current values
        close_price = close_12h[i]
        high_price = high_12h[i]
        low_price = low_12h[i]
        volume_1d_current = df_1d['volume'].values[i] if i < len(df_1d) else volume_1d[-1]
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        volume_spike = volume_1d_current > 2.0 * volume_ma_aligned[i]
        
        # Trend filter
        weekly_uptrend = close_12h[i] > ema_50_aligned[i]
        weekly_downtrend = close_12h[i] < ema_50_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = high_price > donchian_high[i]
        breakout_down = low_price < donchian_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up + volume spike + weekly uptrend
            if breakout_up and volume_spike and weekly_uptrend:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short: Donchian breakout down + volume spike + weekly downtrend
            elif breakout_down and volume_spike and weekly_downtrend:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_12h[i]
                # Exit also on opposite Donchian break
                opposite_break = low_price < donchian_low[i]
                exit_condition = trailing_stop or opposite_break
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_12h[i]
                # Exit also on opposite Donchian break
                opposite_break = high_price > donchian_high[i]
                exit_condition = trailing_stop or opposite_break
            
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