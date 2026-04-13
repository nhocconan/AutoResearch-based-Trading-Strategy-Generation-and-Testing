#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation
    # Uses 1d HTF for trend direction (price vs EMA50) and volume spike confirmation
    # Enters on breakout of Camarilla R4/S4 levels from prior 1d session
    # Works in bull/bear: trend filter ensures we trade with higher timeframe momentum
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, trend filter, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for prior 1d session
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1 / 2
    # S4 = PP - (H - L) * 1.1 / 2
    pp_1d = (high_1d + low_1d + close_1d) / 3
    r4_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2
    s4_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 6h primary timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate 6h ATR for stoploss
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_multiplier = 2.5
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Get 1d bar index for current 6h bar (each 1d bar = 4 6h bars)
        idx_1d = i // 4
        if idx_1d >= len(high_1d):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d[idx_1d] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close_1d[idx_1d] > ema50_1d_aligned[i]
        price_below_ema = close_1d[idx_1d] < ema50_1d_aligned[i]
        
        # Breakout conditions: close breaks Camarilla R4/S4 levels
        breakout_long = close[i] > r4_1d_aligned[i]
        breakout_short = close[i] < s4_1d_aligned[i]
        
        # Entry conditions
        enter_long = breakout_long and price_above_ema and volume_confirmed
        enter_short = breakout_short and price_below_ema and volume_confirmed
        
        # Stoploss conditions
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - atr_multiplier * atr_6h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + atr_multiplier * atr_6h[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0