#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA trend filter and volume confirmation
# - Primary timeframe: 12h for lower trade frequency (target: 12-37 trades/year)
# - Entry: Long when price breaks above 20-period Donchian high AND price > 1w EMA50 (uptrend)
#          Short when price breaks below 20-period Donchian low AND price < 1w EMA50 (downtrend)
# - Volume confirmation: volume > 1.5 * 20-period volume average to filter weak breakouts
# - Exit: ATR-based trailing stop (2.5 * ATR) or opposite Donchian breakout
# - Position sizing: 0.30 (30% of capital) to balance risk and return
# - Uses 1w HTF for trend filter to avoid whipsaws in ranging markets
# - Designed to work in both bull (breakouts with trend) and bear (breakdowns with trend) markets

name = "12h_1w_donchian_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 12h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: rolling max of high
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR(14) for stoploss
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
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry for trailing stop
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: ATR trailing stop or opposite Donchian breakout
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < donch_low[i]:  # Opposite Donchian breakout
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Update lowest low since entry for trailing stop
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: ATR trailing stop or opposite Donchian breakout
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > donch_high[i]:  # Opposite Donchian breakout
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Look for breakout entries with volume confirmation and trend filter
            if (close[i] > donch_high[i] and 
                close[i] > ema_50_1w_aligned[i] and  # Uptrend filter
                volume_confirm[i]):
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.30
            elif (close[i] < donch_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and  # Downtrend filter
                  volume_confirm[i]):
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.30
    
    return signals