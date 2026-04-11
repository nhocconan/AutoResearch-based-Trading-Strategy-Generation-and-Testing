#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ATR stoploss
# - Long: Price breaks above Camarilla H3 level (1d) + volume > 1.3x 20-period 4h average volume
# - Short: Price breaks below Camarilla L3 level (1d) + volume > 1.3x 20-period 4h average volume
# - Exit: ATR-based trailing stop (2.5 ATR from extreme) or opposite Camarilla level touch
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Camarilla pivots from 1d provide institutional support/resistance levels that work in all regimes
# - Volume confirmation filters false breakouts
# - ATR stoploss adapts to volatility
# - Target: 20-50 trades/year to stay within fee drag limits

name = "4h_1d_camarilla_pivot_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    camarilla_h4 = np.zeros_like(close_1d)
    camarilla_l4 = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        rng = high_1d[i] - low_1d[i]
        camarilla_h3[i] = close_1d[i] + rng * 1.1 / 6
        camarilla_l3[i] = close_1d[i] - rng * 1.1 / 6
        camarilla_h4[i] = close_1d[i] + rng * 1.1 / 2
        camarilla_l4[i] = close_1d[i] - rng * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR for stoploss (4h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        h4 = h4_aligned[i]
        l4 = l4_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above H3 with volume confirmation
        if close_price > h3 and vol_confirm:
            enter_long = True
        
        # Short breakout: price closes below L3 with volume confirmation
        if close_price < l3 and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price hits ATR stoploss or breaks below L3
            exit_long = (close_price <= long_stop) or (close_price < l3)
        elif position == -1:
            # Exit short if price hits ATR stoploss or breaks above H3
            exit_short = (close_price >= short_stop) or (close_price > h3)
        
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
            long_stop = max(long_stop, high[i] - 2.5 * atr_14[i])
        elif position == -1:
            # Trail short stop downward: min of current stop and (low + 2.5*ATR)
            short_stop = min(short_stop, low[i] + 2.5 * atr_14[i])
        
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