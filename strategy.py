#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v2"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate daily OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3, L3 (tighter than H4/L4 for more entries)
    hl_range = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * hl_range / 4
    camarilla_l3 = close_1d - 1.1 * hl_range / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: 20-period average on daily volume
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: 34-period EMA on price to avoid counter-trend trades
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_34[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Trend filter: only trade in direction of 34 EMA
        above_ema = price_close > ema_34[i]
        below_ema = price_close < ema_34[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Camarilla H3 + volume confirmation + above EMA34
        if price_close > camarilla_h3_aligned[i] and vol_confirm and above_ema:
            enter_long = True
        
        # Short: Price breaks below Camarilla L3 + volume confirmation + below EMA34
        if price_close < camarilla_l3_aligned[i] and vol_confirm and below_ema:
            enter_short = True
        
        # Exit conditions: price returns to the day's close (pivot point)
        prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
        prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
        exit_long = price_close < prev_close_aligned[i]
        exit_short = price_close > prev_close_aligned[i]
        
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

# Hypothesis: Camarilla H3/L3 breakout on daily timeframe with volume confirmation and EMA34 trend filter.
# Uses tighter Camarilla H3/L3 levels (Close ± 1.1*(High-Low)/4) for more frequent but still filtered entries.
# Works in both bull (breakouts above H3) and bear (breakdowns below L3) by capturing institutional activity.
# Volume confirmation (>1.5x 20-day average) ensures participation. EMA34 filter avoids counter-trend trades.
# Position size 0.25 balances risk and return. Target: 30-50 trades/year to minimize fee drag.