#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band squeeze breakout with weekly trend filter and volume confirmation
# - Bollinger Band squeeze: BB Width at 20-day low indicates low volatility primed for breakout
# - Breakout direction: price breaks above/below upper/lower BB with volume confirmation
# - Weekly trend filter: only take breakouts in direction of weekly EMA50 trend
# - Volume confirmation: breakout volume > 1.5x 20-day average volume
# - ATR(14) trailing stop (2.5x) on 1d timeframe
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 10-20 trades/year (40-80 total over 4 years) to stay within HARD MAX: 150 total

name = "1d_1w_bb_squeeze_breakout_volume_v1"
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
    
    # Pre-compute 1d Bollinger Bands (20, 2)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period SMA and standard deviation
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    
    # Upper and lower Bollinger Bands
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Bollinger Band Width (normalized by SMA)
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # 20-period rolling minimum of BB Width for squeeze detection
    bb_width_min_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    
    # Bollinger Band squeeze condition: current width at 20-period low
    squeeze = bb_width <= bb_width_min_20 * 1.01  # Allow small floating point tolerance
    
    # Pre-compute 1d volume and its 20-period moving average for volume confirmation
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1d ATR(14) for trailing stop
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
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current values
        close_price = close[i]
        volume_current = volume[i]
        weekly_uptrend = close_price > ema_50_aligned[i]
        weekly_downtrend = close_price < ema_50_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_spike = volume_current > 1.5 * volume_ma_20[i]
        
        if position == 0:  # Flat - look for new entries
            # Look for Bollinger Band breakout with squeeze and volume confirmation
            if squeeze[i] and volume_spike:
                # Long breakout: price closes above upper BB in weekly uptrend
                if close_price > upper_bb[i] and weekly_uptrend:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else close_price
                    highest_since_entry = high[i]
                    signals[i] = 0.25
                # Short breakout: price closes below lower BB in weekly downtrend
                elif close_price < lower_bb[i] and weekly_downtrend:
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else close_price
                    lowest_since_entry = low[i]
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, high[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = close_price < highest_since_entry - 2.5 * atr[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, low[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = close_price > lowest_since_entry + 2.5 * atr[i]
                exit_condition = trailing_stop
            
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