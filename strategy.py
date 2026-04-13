#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike filter
    # Works in bull/bear: Camarilla captures key intraday levels, volume confirms institutional participation
    # Target: 40-80 trades/year to balance edge and fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Camarilla calculation (requires daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get daily OHLC
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate Camarilla levels for each day (based on previous day's range)
    # Camarilla H4 = Close + (High - Low) * 1.1 / 2
    # Camarilla L4 = Close - (High - Low) * 1.1 / 2
    camarilla_h4 = daily_close + (daily_high - daily_low) * 1.1 / 2
    camarilla_l4 = daily_close - (daily_high - daily_low) * 1.1 / 2
    
    # Calculate 1d volume average (20-period) for spike detection
    vol_avg_20_1d = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h primary timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for exit logic
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get corresponding 1d index for volume check
        idx_1d = i // 6  # 6 * 4h bars = 1 day
        if idx_1d >= len(daily_volume):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = daily_volume[idx_1d] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Entry conditions: Camarilla level break + volume confirmation
        enter_long = (close[i] > camarilla_h4_aligned[i]) and volume_confirmed
        enter_short = (close[i] < camarilla_l4_aligned[i]) and volume_confirmed
        
        # Exit conditions: reverse signal or volume deterioration
        exit_long = position == 1 and (close[i] < camarilla_l4_aligned[i] or not volume_confirmed)
        exit_short = position == -1 and (close[i] > camarilla_h4_aligned[i] or not volume_confirmed)
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]  # record entry price at close (filled next bar open)
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]  # record entry price at close (filled next bar open)
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

name = "4h_1d_camarilla_pivot_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0