#!/usr/bin/env python3
"""
Experiment #174: 1h 4h/1d Donchian Breakout with Volume and Session Filter
HYPOTHESIS: Use 4h for trend direction (price > Donchian(20) high/low) and 1d for regime filter (ADX > 25), 
then enter on 1h breakouts in direction of trend during active session (08-20 UTC). 
Volume confirmation (>1.5x average) reduces false breakouts. 
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_174_1h_donchian_4h_1d_trend_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for trend direction (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high_20_4h = rolling_max(high_4h, 20)
    donchian_low_20_4h = rolling_min(low_4h, 20)
    
    # Trend direction: price above/below Donchian channels
    trend_up_4h = close_4h > donchian_high_20_4h
    trend_down_4h = close_4h < donchian_low_20_4h
    
    # Align 4h trend to 1h timeframe
    trend_up_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_up_20_4h)
    trend_down_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_down_20_4h)
    
    # === HTF: 1d data for regime filter (ADX > 25 = trending market) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, window=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        def wilders_smooth(data, period):
            res = np.full_like(data, np.nan)
            for i in range(len(data)):
                if i < period:
                    continue
                if i == period:
                    res[i] = np.nansum(data[i-period+1:i+1])
                else:
                    res[i] = res[i-1] - (res[i-1]/period) + data[i]
            return res
        
        tr_smoothed = wilders_smooth(tr, window)
        plus_dm_smoothed = wilders_smooth(plus_dm, window)
        minus_dm_smoothed = wilders_smooth(minus_dm, window)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smoothed / tr_smoothed
        minus_di = 100 * minus_dm_smoothed / tr_smoothed
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                      0.0)
        adx = wilders_smooth(dx, window)
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    trending_regime = adx_14_1d > 25
    
    # Align 1d regime to 1h timeframe
    trending_regime_aligned = align_htf_to_ltf(prices, df_1d, trending_regime)
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr_1h = np.zeros(n)
    tr_1h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_1h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # Enough for 4h Donchian (20*4=80) + 1d ADX + volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(trend_up_4h_aligned[i]) or np.isnan(trend_down_4h_aligned[i]) or
            np.isnan(trending_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require: in session, volume spike, trending regime (from 1d ADX)
        volume_spike = vol_ratio[i] > 1.5
        session_ok = in_session[i]
        regime_ok = trending_regime_aligned[i]
        
        if not (session_ok and volume_spike and regime_ok):
            signals[i] = 0.0
            continue
        
        # Long: 4h uptrend (price > 4h Donchian high) and 1h breakout above recent high
        if trend_up_4h_aligned[i]:
            # 1h Donchian breakout (20-period) for entry timing
            lookback = min(20, i)
            if lookback >= 20:
                recent_high = np.max(high[i-lookback:i])
                if price > recent_high:
                    in_position = True
                    position_side = 1
                    entry_price = price
                    bars_since_entry = 0
                    signals[i] = SIZE
        # Short: 4h downtrend (price < 4h Donchian low) and 1h breakout below recent low
        elif trend_down_4h_aligned[i]:
            lookback = min(20, i)
            if lookback >= 20:
                recent_low = np.min(low[i-lookback:i])
                if price < recent_low:
                    in_position = True
                    position_side = -1
                    entry_price = price
                    bars_since_entry = 0
                    signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals