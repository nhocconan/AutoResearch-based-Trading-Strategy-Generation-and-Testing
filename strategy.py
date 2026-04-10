#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA100 trend filter and volume confirmation
# - Uses Donchian(20) from 4h for breakout levels (structure-based)
# - 1d EMA100 trend filter ensures trades align with longer-term trend (adapts to bull/bear)
# - Volume confirmation: current volume > 2.0x 20-period average to filter weak breakouts
# - Exit: touch of opposite Donchian level or ATR-based stoploss
# - Position size: 0.25 (25% of capital) to balance risk and minimize fee drag
# - Target: 20-50 trades/year on 4h (80-200 total over 4 years) to stay within trade limits
# - Works in bull/bear: EMA100 reacts to regime changes, volume reduces false signals, Donchian provides objective breakout levels

name = "4h_1d_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d EMA100 for trend filter
    close_1d = df_1d['close'].values
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    trend_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate rolling max/min for Donchian channels
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels (already in 4h timeframe, no alignment needed)
    # But we need to shift by 1 to avoid look-ahead (use previous bar's breakout level)
    donchian_high = np.roll(high_roll_max, 1)
    donchian_low = np.roll(low_roll_min, 1)
    donchian_high[0] = np.nan  # First value has no previous bar
    donchian_low[0] = np.nan
    
    # Pre-compute 4h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR for stoploss (14-period)
    tr1 = pd.Series(prices['high']).rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0]).values
    tr2 = pd.Series(prices['high']).rolling(2).apply(lambda x: abs(x.iloc[1] - prices['close'].iloc[x.index[0]])).values
    tr3 = pd.Series(prices['low']).rolling(2).apply(lambda x: abs(x.iloc[0] - prices['close'].iloc[x.index[0]])).values
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):  # Start after warmup for Donchian and EMA100
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(trend_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma_20[i]
        
        # Get current 1d close for trend filter (aligned)
        close_1d_current = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_current)
        
        # 1d trend filter: price > EMA100 = bullish, price < EMA100 = bearish
        bullish_trend = not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1d_aligned[i] > trend_aligned[i]
        bearish_trend = not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1d_aligned[i] < trend_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > Donchian High AND bullish trend AND volume confirmation
            if prices['close'].iloc[i] > donchian_high[i] and bullish_trend and volume_confirm:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = 0.25
            # Short conditions: price < Donchian Low AND bearish trend AND volume confirmation
            elif prices['close'].iloc[i] < donchian_low[i] and bearish_trend and volume_confirm:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Exit conditions: price touches opposite Donchian level
            exit_long = prices['close'].iloc[i] < donchian_low[i]   # Price breaks below Donchian Low (exit long)
            exit_short = prices['close'].iloc[i] > donchian_high[i]  # Price breaks above Donchian High (exit short)
            
            # Stoploss conditions: ATR-based (2 * ATR)
            if position == 1:
                stoploss_hit = prices['close'].iloc[i] < entry_price - 2.0 * atr[i]
            else:  # position == -1
                stoploss_hit = prices['close'].iloc[i] > entry_price + 2.0 * atr[i]
            
            exit_condition = exit_long or exit_short or stoploss_hit
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals