#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter, volume confirmation, and ATR stoploss
# Long when price breaks above Donchian upper channel AND 1w EMA50 uptrend AND volume > 2.0 * 20-period avg volume
# Short when price breaks below Donchian lower channel AND 1w EMA50 downtrend AND volume > 2.0 * 20-period avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.5 * ATR OR short and price > lowest_low + 2.5 * ATR
# Uses discrete sizing 0.30 to balance risk and return (BTC -77% in 2022 → ~23.1% loss at 0.30 exposure)
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Donchian channels provide clear structure, 1w EMA50 filters primary trend, high volume threshold confirms breakout strength

name = "1d_Donchian20_1wEMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(10) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # Align HTF indicators to 1d timeframe (wait for completed HTF bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(atr_10[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            breakout_up = close[i] > highest_high_20[i-1]  # Break above previous upper channel
            breakout_down = close[i] < lowest_low_20[i-1]  # Break below previous lower channel
            
            # Long: breakout above upper channel AND uptrend AND volume spike
            if breakout_up and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
                highest_high_since_entry = close[i]
            # Short: breakout below lower channel AND downtrend AND volume spike
            elif breakout_down and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
                lowest_low_since_entry = close[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, close[i])
            # Exit long: price drops below highest_high - 2.5 * ATR (trailing stop)
            if close[i] < highest_high_since_entry - 2.5 * atr_10[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, close[i])
            # Exit short: price rises above lowest_low + 2.5 * ATR (trailing stop)
            if close[i] > lowest_low_since_entry + 2.5 * atr_10[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.30
    
    return signals