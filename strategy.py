#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and ATR trailing stop
# - Uses Donchian(20) channels on 4h for breakout entries (long above upper channel, short below lower)
# - Requires 1d volume > 1.5 * 20-day volume average for confirmation (reduces false breakouts)
# - Uses ATR(14) for dynamic trailing stoploss (2.5 * ATR from extreme) and position sizing (0.25)
# - Works in bull markets via breakouts above resistance, in bear via breakdowns below support
# - Target: 20-35 trades/year on 4h timeframe (80-140 total over 4 years) to avoid fee drag
# - Donchian channels provide objective volatility-based support/resistance levels

name = "4h_1d_donchian_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume confirmation: volume > 1.5 * 20-day average
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper channel: highest high over last 20 periods
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low over last 20 periods
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for stoploss and sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry for trailing stop
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: ATR trailing stop or mean reversion to midpoint
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR trailing stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < (dc_upper[i] + dc_lower[i]) / 2:  # Mean reversion exit to midpoint
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry for trailing stop
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: ATR trailing stop or mean reversion to midpoint
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:  # ATR trailing stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > (dc_upper[i] + dc_lower[i]) / 2:  # Mean reversion exit to midpoint
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries with volume confirmation
            if close[i] > dc_upper[i] and volume_confirm_1d_aligned[i]:  # Break above upper channel
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif close[i] < dc_lower[i] and volume_confirm_1d_aligned[i]:  # Break below lower channel
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals