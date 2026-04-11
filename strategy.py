#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_volatility_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # 12h ATR for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Bollinger Bands for mean reversion filter
    close_1d = df_1d['close'].values
    bb_mid = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + (bb_std * 2.0)
    bb_lower = bb_mid - (bb_std * 2.0)
    bb_width = bb_upper - bb_lower
    
    # 6h Donchian channels for breakout
    high_6h = high
    low_6h = low
    donch_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 6h volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Align 12h ATR to 6h
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Align 1d Bollinger Bands to 6h
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(bb_width_aligned[i]) or
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volatility filter: 12h ATR > 0.5 * 60-period average ATR (avoid low volatility)
        atr_ma_60 = pd.Series(atr_12h_aligned).rolling(window=60, min_periods=60).mean().values
        vol_filter = atr_12h_aligned[i] > 0.5 * atr_ma_60[i] if not np.isnan(atr_ma_60[i]) else False
        
        # Bollinger Band filter: BB width > 10th percentile (avoid squeeze)
        bb_width_ma_50 = pd.Series(bb_width_aligned).rolling(window=50, min_periods=50).mean().values
        bb_filter = bb_width_aligned[i] > 0.5 * bb_width_ma_50[i] if not np.isnan(bb_width_ma_50[i]) else False
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_50[i]
        
        # Breakout conditions
        breakout_up = price_close > donch_high[i]
        breakout_down = price_close < donch_low[i]
        
        # Entry conditions: breakout with volatility, BB width, and volume filters
        enter_long = False
        enter_short = False
        
        if breakout_up and vol_filter and bb_filter and vol_confirm:
            enter_long = True
        
        if breakout_down and vol_filter and bb_filter and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Donchian level
        exit_long = price_close < donch_low[i]
        exit_short = price_close > donch_high[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6s volatility breakout strategy using 12h ATR filter, 1d Bollinger Band width filter, and 6h Donchian breakouts.
# Enters long on Donchian high breakout with elevated volatility (12h ATR > 50% of MA), 
# sufficient Bollinger Band width (>50% of MA to avoid squeezes), and volume confirmation (>1.5x 50-period avg).
# Enters short on Donchian low breakout under same conditions.
# Exits when price returns to opposite Donchian level.
# Designed to capture strong moves in both bull and bear markets while avoiding choppy, low-volatility periods.
# Position size 0.25 limits risk. Target: 20-50 trades per year to minimize fee drag.
# Uses multi-timeframe alignment to ensure no look-ahead bias.