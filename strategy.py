#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot breakouts with 1d volume filter and session filter (08-20 UTC)
# - Long: price breaks above 4h Camarilla H3 level, 1d volume > 1.5x 20-period average, within active session
# - Short: price breaks below 4h Camarilla L3 level, 1d volume > 1.5x 20-period average, within active session
# - Exit: price returns to 4h Camarilla H4/L4 levels or ATR-based stop (2x ATR)
# - Uses discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# - Uses 4h for signal direction (structure), 1h only for entry timing precision
# - Session filter reduces noise trades during low-volume off-hours
# - Volume confirmation ensures breakouts have conviction

name = "1h_4h_1d_camarilla_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return signals
    
    # Pre-compute 4h Camarilla levels (based on previous 4h bar's OHLC)
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    camarilla_h4 = close_4h + (high_4h - low_4h) * 1.1 / 2
    camarilla_h3 = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_l3 = close_4h - (high_4h - low_4h) * 1.1 / 4
    camarilla_l4 = close_4h - (high_4h - low_4h) * 1.1 / 2
    
    # Align 4h Camarilla levels to 1h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute ATR for stoploss (1h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or
            np.isnan(l4_aligned[i]) or np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside active trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        h4_level = h4_aligned[i]
        l4_level = l4_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Camarilla H3, volume confirmation
        if close_price > h3_level and vol_confirm:
            enter_long = True
        
        # Short breakout: price below Camarilla L3, volume confirmation
        if close_price < l3_level and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches H4 or ATR-based stop
            exit_long = (close_price >= h4_level) or (close_price <= entry_price - 2.0 * atr_14[i])
        elif position == -1:
            # Exit short if price reaches L4 or ATR-based stop
            exit_short = (close_price <= l4_level) or (close_price >= entry_price + 2.0 * atr_14[i])
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals