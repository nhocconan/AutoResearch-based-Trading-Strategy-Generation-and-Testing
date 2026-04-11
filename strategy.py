#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w volume confirmation and ATR trailing stop
# - Long: Price breaks above Camarilla H4 level (resistance) + 1w volume > 1.5x 20-period average
# - Short: Price breaks below Camarilla L4 level (support) + 1w volume > 1.5x 20-period average
# - Exit: ATR-based trailing stop (2.5 ATR from extreme) or opposite Camarilla level break
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) for 12h timeframe
# - Camarilla pivots derived from 1d OHLC provide institutional reversal levels effective in all regimes
# - Volume confirmation on 1w filters low-conviction breakouts
# - ATR stoploss adapts to volatility while respecting engine semantics

name = "12h_1w_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate Camarilla levels from 1d OHLC (previous day)
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # We focus on H4/L4 for breakouts
    camarilla_h4 = df_1d['close'] + 1.5 * (df_1d['high'] - df_1d['low'])
    camarilla_l4 = df_1d['close'] - 1.5 * (df_1d['high'] - df_1d['low'])
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 for completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4.values)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4.values)
    
    # Load 1w data ONCE before loop for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w volume confirmation (20-period average)
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    
    # Pre-compute ATR for stoploss (12h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        high_price = high[i]
        low_price = low[i]
        
        # Camarilla levels from previous 1d bar
        h4_level = camarilla_h4_aligned[i]
        l4_level = camarilla_l4_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average on 1w
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above H4 resistance with volume confirmation
        if close_price > h4_level and vol_confirm:
            enter_long = True
        
        # Short breakout: price closes below L4 support with volume confirmation
        if close_price < l4_level and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price hits ATR stoploss or breaks below L4 support
            exit_long = (close_price <= long_stop) or (close_price < l4_level)
        elif position == -1:
            # Exit short if price hits ATR stoploss or breaks above H4 resistance
            exit_short = (close_price >= short_stop) or (close_price > h4_level)
        
        # Update stoploss levels when entering a position
        if enter_long:
            entry_price = close_price
            long_stop = entry_price - 2.5 * atr_14[i]
        elif enter_short:
            entry_price = close_price
            short_stop = entry_price + 2.5 * atr_14[i]
        
        # Update trailing stoploss for existing positions
        if position == 1:
            # Trail long stop upward: max of current stop and (high - 2.5*ATR)
            long_stop = max(long_stop, high_price - 2.5 * atr_14[i])
        elif position == -1:
            # Trail short stop downward: min of current stop and (low + 2.5*ATR)
            short_stop = min(short_stop, low_price + 2.5 * atr_14[i])
        
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