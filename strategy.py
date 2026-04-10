#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + ATR stoploss
# - Donchian(20) breakout captures strong momentum moves in both bull and bear markets
# - 1d volume spike (>2.0x 20-day average of volume/ATR) confirms institutional participation
# - ATR(14) trailing stop (2.0x) manages risk and adapts to volatility
# - Discrete position sizing (0.25) minimizes fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Works in bull/bear: breakouts work in trending markets, volume filter avoids false signals,
#   ATR stop controls drawdown during reversals

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ATR volume for confirmation (14-period ATR)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = np.nan
    tr2_1d[0] = np.nan
    tr3_1d[0] = np.nan
    tr_1d = np.maximum.reduce([tr1_1d, tr2_1d, tr3_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR volume: volume / ATR (normalizes volume by volatility)
    atr_volume_1d = volume_1d / atr_1d
    atr_volume_ma_20_1d = pd.Series(atr_volume_1d).rolling(window=20, min_periods=20).mean().values
    atr_volume_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_volume_ma_20_1d)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR for trailing stop (14-period)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i]) or 
            np.isnan(atr_volume_ma_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 1d ATR volume for filter (aligned)
        atr_volume_1d_current = atr_volume_1d
        atr_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_volume_1d_current)
        
        # Volume confirmation: current 1d ATR volume > 2.0x 20-day average
        volume_confirm = atr_volume_1d_aligned[i] > 2.0 * atr_volume_ma_aligned[i]
        
        # Donchian breakout conditions
        close_price = close_4h[i]
        
        # Long breakout: price > Donchian high (20-period)
        long_breakout = close_price > donchian_high[i]
        
        # Short breakout: price < Donchian low (20-period)
        short_breakout = close_price < donchian_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Long breakout AND volume confirmation
            if long_breakout and volume_confirm:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short conditions: Short breakout AND volume confirmation
            elif short_breakout and volume_confirm:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.0*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.0 * atr[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.0*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.0 * atr[i]
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