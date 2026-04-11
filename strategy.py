#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return signals
    
    # Calculate 12h OHLC for Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: H4, L4 (main resistance/support)
    # Formula: H4 = Close + 1.1 * (High - Low) / 2
    #          L4 = Close - 1.1 * (High - Low) / 2
    hl_range = high_12h - low_12h
    camarilla_h4 = close_12h + 1.1 * hl_range / 2
    camarilla_l4 = close_12h - 1.1 * hl_range / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    # Volume confirmation: 24-period average on 12h (2 days)
    volume_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Trend filter: 50-period EMA on 12h to avoid counter-trend trades
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_sma_24[i]) or np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.8x 24-period average
        vol_confirm = volume_current > 1.8 * volume_sma_24[i]
        
        # Trend filter: only trade in direction of 50 EMA
        above_ema = price_close > ema_50[i]
        below_ema = price_close < ema_50[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Camarilla H4 + volume confirmation + above EMA50
        if price_close > camarilla_h4_aligned[i] and vol_confirm and above_ema:
            enter_long = True
        
        # Short: Price breaks below Camarilla L4 + volume confirmation + below EMA50
        if price_close < camarilla_l4_aligned[i] and vol_confirm and below_ema:
            enter_short = True
        
        # Exit conditions: price returns to the day's close (pivot point)
        # Use previous day's close as exit level
        prev_close_12h = np.concatenate([[np.nan], close_12h[:-1]])
        prev_close_aligned = align_htf_to_ltf(prices, df_12h, prev_close_12h)
        exit_long = price_close < prev_close_aligned[i]
        exit_short = price_close > prev_close_aligned[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla breakout on 12h timeframe with volume confirmation and EMA50 trend filter.
# Uses Camarilla H4/L4 levels (Close ± 1.1*(High-Low)/2) from 12h timeframe for entry.
# Exit when price returns to previous 12h close (pivot point).
# Volume confirmation (>1.8x 24-period average) ensures institutional participation.
# EMA50 filter prevents counter-trend trades, improving win rate in trending markets.
# Works in both bull and breakout scenarios by capturing institutional breakout attempts.
# Position size set to 0.30 to balance risk and return. Target: 25-40 trades/year to minimize fee drag.