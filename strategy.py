#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# - Uses 4h Donchian channel (20-period high/low) for breakout entries
# - 1d EMA50 trend filter: only long when price > EMA50, short when price < EMA50
# - Volume confirmation: volume > 1.5 * 20-period average to avoid false breakouts
# - ATR(14) stoploss: 2.0 * ATR for risk management
# - Position size: 0.25 (25% of capital) to balance risk and return
# - Works in bull markets via upward breakouts, in bear via downward breakouts with trend filter
# - Target: 25-35 trades/year on 4h timeframe (100-140 total over 4 years) to avoid fee drag

name = "4h_1d_donchian_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: highest high over past 20 periods
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over past 20 periods
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for stoploss
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
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: ATR stoploss or Donchian low break
            if close[i] < highest_high_since_entry - 2.0 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < donch_low[i]:  # Break below Donchian low
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: ATR stoploss or Donchian high break
            if close[i] > lowest_low_since_entry + 2.0 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > donch_high[i]:  # Break above Donchian high
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries with volume confirmation and trend filter
            if close[i] > donch_high[i] and volume_confirm[i] and close[i] > ema_50_aligned[i]:  # Break above Donchian high with uptrend
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif close[i] < donch_low[i] and volume_confirm[i] and close[i] < ema_50_aligned[i]:  # Break below Donchian low with downtrend
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals