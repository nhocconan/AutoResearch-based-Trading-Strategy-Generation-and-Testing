#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h EMA crossover (21/55) with 1d Donchian breakout filter and volume confirmation
    # Long: EMA21 crosses above EMA55 + price > 1d Donchian High(20) + volume > 1.5x 20-period average
    # Short: EMA21 crosses below EMA55 + price < 1d Donchian Low(20) + volume > 1.5x 20-period average
    # Uses discrete sizing (0.25) and ATR-based stoploss to manage risk
    # Target: 12-37 trades/year to stay within 6h optimal range (50-150 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate EMA crossover on 6h data
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Calculate volume average for confirmation
    vol_avg_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR for stoploss
    atr_6h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_6h[i] = tr
        else:
            atr_6h[i] = 0.93 * atr_6h[i-1] + 0.07 * tr
    
    for i in range(55, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_21[i]) or
            np.isnan(ema_55[i]) or
            np.isnan(vol_avg_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_6h[i]
        
        # EMA crossover signals
        ema_cross_up = ema_21[i] > ema_55[i] and ema_21[i-1] <= ema_55[i-1]
        ema_cross_down = ema_21[i] < ema_55[i] and ema_21[i-1] >= ema_55[i-1]
        
        # Donchian filter: price must be beyond the 1d Donchian levels
        price_above_donchian_high = close[i] > donchian_high_aligned[i]
        price_below_donchian_low = close[i] < donchian_low_aligned[i]
        
        # Entry conditions
        long_entry = ema_cross_up and price_above_donchian_high and volume_confirmed
        short_entry = ema_cross_down and price_below_donchian_low and volume_confirmed
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_6h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_6h[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif short_entry and position != -1:
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

name = "6h_1d_ema_crossover_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0