#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume spike confirmation
# - Donchian breakout: price breaks above 20-period high (long) or below 20-period low (short)
# - Weekly trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - Volume confirmation: current 1d volume > 2.0x 20-period average to filter weak breakouts
# - ATR(14) trailing stop (2.5x) on 1d timeframe for risk management
# - Discrete position sizing (0.30) to balance return and drawdown
# - Target: 20-40 trades/year (80-160 total over 4 years) within HARD MAX: 150 total
# - Works in bull markets via breakout continuation and in bear markets via short breakdowns
# - Volume spike ensures breakouts have conviction, reducing false signals

name = "1d_1w_donchian_volume_trend_v1"
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
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Rolling max/min for Donchian channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d volume and its 20-period moving average for volume confirmation
    volume_1d = prices['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1d ATR(14) for trailing stop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20_1d[i]) or 
            np.isnan(atr_1d[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_spike = volume_1d[i] > 2.0 * volume_ma_20_1d[i]
        
        # Donchian breakout conditions
        breakout_up = close_1d[i] > donchian_high[i-1]  # Break above previous period's high
        breakout_down = close_1d[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Weekly trend filter
        weekly_uptrend = close_1d[i] > ema_50_aligned[i]
        weekly_downtrend = close_1d[i] < ema_50_aligned[i]
        
        close_price = close_1d[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: bullish breakout + volume spike + weekly uptrend
            if breakout_up and volume_spike and weekly_uptrend:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.30
            # Short entry: bearish breakdown + volume spike + weekly downtrend
            elif breakout_down and volume_spike and weekly_downtrend:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_1d[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_1d[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.30
                else:
                    signals[i] = -0.30
    
    return signals