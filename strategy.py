#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h strategy using 1d Camarilla pivot levels (H3/L3) with 1w EMA trend filter and volume confirmation
    # Works in both bull and bear: Pivot levels provide mean-reversion entries in ranging markets,
    # 1w EMA > price for shorts and < price for longs filters trend direction,
    # Volume confirmation ensures participation. Discrete sizing (0.25) minimizes fee drag.
    # Target: 20-40 trades/year to stay within 4h optimal range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Camarilla pivot levels (H3, L3, H4, L4)
    # Based on previous day's OHLC
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    camarilla_h4 = np.zeros_like(close_1d)
    camarilla_l4 = np.zeros_like(close_1d)
    pivot = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC for today's levels
        high_y = high_1d[i-1]
        low_y = low_1d[i-1]
        close_y = close_1d[i-1]
        
        pivot[i] = (high_y + low_y + close_y) / 3.0
        range_y = high_y - low_y
        
        camarilla_h3[i] = close_y + range_y * 1.1 / 4.0
        camarilla_l3[i] = close_y - range_y * 1.1 / 4.0
        camarilla_h4[i] = close_y + range_y * 1.1 / 2.0
        camarilla_l4[i] = close_y - range_y * 1.1 / 2.0
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d volume for confirmation (20-period average)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h primary timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.2x 20-period average
        idx_1d = i // (24 * 6)  # 1d bars in 4h timeframe (6 bars per day)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.2 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: price relative to 1w EMA50
        above_ema = close[i] > ema_50_1w_aligned[i]
        below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions: Camarilla H3/L3 touch with trend filter and volume
        enter_long = (close[i] <= camarilla_h3_aligned[i] * 1.002) and below_ema and volume_confirmed
        enter_short = (close[i] >= camarilla_l3_aligned[i] * 0.998) and above_ema and volume_confirmed
        
        # Stoploss: at opposite Camarilla level (H4 for longs, L4 for shorts)
        stoploss_long = camarilla_l4_aligned[i]
        stoploss_short = camarilla_h4_aligned[i]
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < stoploss_long
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > stoploss_short
        
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

name = "4h_1d_1w_camarilla_pivot_ema_volume_v1"
timeframe = "4h"
leverage = 1.0