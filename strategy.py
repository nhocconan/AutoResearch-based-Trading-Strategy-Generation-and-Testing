#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50 > EMA200) and volume confirmation
# - Long: Close breaks above Donchian(20) high + 1d EMA50 > EMA200 (uptrend) + 1d volume > 1.5x 20-period MA
# - Short: Close breaks below Donchian(20) low + 1d EMA50 < EMA200 (downtrend) + 1d volume > 1.5x 20-period MA
# - Exit: Opposite Donchian breakout or ATR-based stoploss (2x ATR)
# - Position sizing: 0.30 (discrete level)
# - Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag
# - Donchian breakouts capture strong momentum moves; 1d EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation reduces false signals in ranging markets

name = "4h_1d_donchian_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) channels for 4h
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA(50) and EMA(200) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate ATR(14) for 4h for stoploss
    tr1 = pd.Series(high_4h - low_4h).values
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1))).values
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1))).values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Set first ATR value to avoid NaN
    atr_14[0] = tr1[0] if not np.isnan(tr1[0]) else 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(200, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get aligned 1d data for current 4h bar (completed 1d bar)
        ema_50_current = ema_50_aligned[i]
        ema_200_current = ema_200_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Trend condition: EMA(50) > EMA(200) for uptrend, EMA(50) < EMA(200) for downtrend
        uptrend = ema_50_current > ema_200_current
        downtrend = ema_50_current < ema_200_current
        
        # Volume spike condition: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d_current > 1.5 * volume_ma_current
        
        # Donchian breakout conditions
        donchian_breakout_up = close_price > highest_high_20[i-1]  # Break above previous period's high
        donchian_breakout_down = close_price < lowest_low_20[i-1]  # Break below previous period's low
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Donchian breakout up + uptrend + volume spike
            if (donchian_breakout_up and uptrend and volume_spike):
                position = 1
                entry_price = close_price
                atr_stop = entry_price - 2.0 * atr_14[i]
                signals[i] = 0.30
            # Short entry: Donchian breakout down + downtrend + volume spike
            elif (donchian_breakout_down and downtrend and volume_spike):
                position = -1
                entry_price = close_price
                atr_stop = entry_price + 2.0 * atr_14[i]
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update trailing stop for long positions
            if position == 1:
                # Update stop to trail behind price
                new_stop = close_price - 2.0 * atr_14[i]
                if new_stop > atr_stop:
                    atr_stop = new_stop
                
                # Exit conditions: opposite Donchian breakout or stoploss hit
                if (donchian_breakout_down or close_price <= atr_stop):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                # Update stop to trail behind price (for shorts, stop moves down)
                new_stop = close_price + 2.0 * atr_14[i]
                if new_stop < atr_stop:
                    atr_stop = new_stop
                
                # Exit conditions: opposite Donchian breakout or stoploss hit
                if (donchian_breakout_up or close_price >= atr_stop):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals