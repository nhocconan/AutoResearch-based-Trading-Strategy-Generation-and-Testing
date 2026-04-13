#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with volume confirmation and ATR filter
    # Long: price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND ATR(14) > 0
    # Short: price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND ATR(14) > 0
    # Exit: opposite Donchian break or volume dry-up
    # Using 4h for signal direction and 12h for trend filter (EMA50)
    # Discrete position sizing (0.25) to minimize fee churn
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h (wait for completed 4h bar)
    high_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        ema_12h = np.full(len(close_4h), np.nan)
    else:
        close_12h = df_12h['close'].values
        ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h) if len(df_12h) >= 50 else np.full(n, np.nan)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # ATR filter: only trade when ATR(14) > 0 (avoid dead markets)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_filter = atr_14 > 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr_filter[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 12h EMA50, only short if price < 12h EMA50
        long_trend_ok = True
        short_trend_ok = True
        if not np.isnan(ema_12h_aligned[i]):
            long_trend_ok = close[i] > ema_12h_aligned[i]
            short_trend_ok = close[i] < ema_12h_aligned[i]
        
        # ATR filter
        atr_ok = atr_filter[i]
        
        # Entry logic: Donchian breakout + volume + trend + ATR
        long_entry = (close[i] > high_20_aligned[i]) and vol_confirm and long_trend_ok and atr_ok
        short_entry = (close[i] < low_20_aligned[i]) and vol_confirm and short_trend_ok and atr_ok
        
        # Exit logic: opposite Donchian break or volume dry-up
        long_exit = (close[i] < low_20_aligned[i]) or not vol_confirm
        short_exit = (close[i] > high_20_aligned[i]) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0