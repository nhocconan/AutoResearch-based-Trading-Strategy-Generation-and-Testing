#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter (EMA50 > EMA200) and volume confirmation
# - Long: Price breaks above Donchian(20) high + EMA50 > EMA200 (uptrend) + volume > 1.5x 20-period MA
# - Short: Price breaks below Donchian(20) low + EMA50 < EMA200 (downtrend) + volume > 1.5x 20-period MA
# - Exit: Opposite Donchian breakout or trailing stop (signal=0 when price < highest - 2*ATR for longs, price > lowest + 2*ATR for shorts)
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag
# - Donchian breakouts capture strong momentum moves; 1d EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation reduces false breakouts in low-participation markets

name = "12h_1d_donchian_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period) for 12h
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for 12h (for stoploss)
    tr1 = pd.Series(high_12h - low_12h).values
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1))).values
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1))).values
    tr2[0] = 0  # First bar has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA(50) and EMA(200) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 12h volume moving average (20-period)
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # Track entry price for stoploss
    highest_since_entry = 0.0  # Track highest high since entry for trailing stop
    lowest_since_entry = 0.0   # Track lowest low since entry for trailing stop
    
    for i in range(200, n):  # Start after warmup period (need at least 200 for EMA200)
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_ma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h data
        close_price = close_12h[i]
        high_price = high_12h[i]
        low_price = low_12h[i]
        volume = volume_12h[i]
        
        # Get aligned 1d data for current 12h bar (completed 1d bar)
        ema_50_current = ema_50_aligned[i]
        ema_200_current = ema_200_aligned[i]
        
        # Trend condition: EMA(50) > EMA(200) for uptrend, EMA(50) < EMA(200) for downtrend
        uptrend = ema_50_current > ema_200_current
        downtrend = ema_50_current < ema_200_current
        
        # Volume spike condition: current volume > 1.5x 20-period MA
        volume_spike = volume > 1.5 * volume_ma_20_12h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high + uptrend + volume spike
            if (close_price > highest_high[i] and uptrend and volume_spike):
                position = 1
                entry_price = close_price
                highest_since_entry = high_price
                lowest_since_entry = low_price
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low + downtrend + volume spike
            elif (close_price < lowest_low[i] and downtrend and volume_spike):
                position = -1
                entry_price = close_price
                highest_since_entry = high_price
                lowest_since_entry = low_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, high_price)
                lowest_since_entry = min(lowest_since_entry, low_price)
            else:  # position == -1
                highest_since_entry = max(highest_since_entry, high_price)
                lowest_since_entry = min(lowest_since_entry, low_price)
            
            # Exit conditions:
            # 1. Opposite Donchian breakout
            # 2. Trailing stop: price < highest - 2*ATR for longs, price > lowest + 2*ATR for shorts
            exit_signal = False
            
            if position == 1:
                # Exit long if price breaks below Donchian low or hits trailing stop
                if (close_price < lowest_low[i] or 
                    close_price < highest_since_entry - 2.0 * atr[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short if price breaks above Donchian high or hits trailing stop
                if (close_price > highest_high[i] or 
                    close_price > lowest_since_entry + 2.0 * atr[i]):
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals