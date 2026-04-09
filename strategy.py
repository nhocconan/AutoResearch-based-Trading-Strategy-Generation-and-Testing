#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR volatility filter and volume confirmation
# - Uses 4h Donchian(20) breakouts for entries (long on upper channel, short on lower)
# - Requires 1d ATR(14) > 20-period 1d ATR MA to confirm sufficient volatility (avoids chop)
# - Requires volume > 1.5 * 20-period 4h volume average for confirmation
# - Uses ATR-based trailing stop (2.5 * ATR) to manage risk
# - Position size: 0.25 (25% of capital) to limit drawdown
# - Works in trending markets via breakouts, avoids ranging markets via volatility filter
# - Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years) to minimize fee drag

name = "4h_1d_donchian_breakout_atr_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ATR(14) and its 20-period moving average for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr_1d > atr_ma_1d  # Only trade when volatility is above average
    
    # Align 1d volatility filter to 4h timeframe
    volatility_filter_aligned = align_htf_to_ltf(prices, df_1d, volatility_filter)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper channel: highest high over past 20 periods
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low over past 20 periods
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for stoploss and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volatility_filter_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: trailing stoploss or Donchian re-entry
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR trailing stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < upper_channel[i]:  # Exit if price breaks below upper channel
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: trailing stoploss or Donchian re-entry
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:  # ATR trailing stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > lower_channel[i]:  # Exit if price breaks above lower channel
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries with volatility and volume confirmation
            if (close[i] > upper_channel[i] and 
                volatility_filter_aligned[i] and 
                volume_confirm[i]):
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif (close[i] < lower_channel[i] and 
                  volatility_filter_aligned[i] and 
                  volume_confirm[i]):
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals