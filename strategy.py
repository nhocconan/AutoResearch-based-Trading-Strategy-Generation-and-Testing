#!/usr/bin/env python3
"""
Experiment #814: 1h Donchian(20) + 4h/1d Trend + Volume Spike + Session Filter
HYPOTHESIS: Donchian breakouts on 1h capture momentum, filtered by 4h/1d trend alignment 
and volume confirmation (>2.0x average). Long when price breaks above Donchian upper 
AND 4h/1d trend up AND volume spike AND session active. Short when price breaks below 
Donchian lower AND 4h/1d trend down AND volume spike AND session active. Uses session 
filter (08-20 UTC) to reduce noise. Discrete position sizing (0.20) to minimize fee 
drag. Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_814_1h_donchian20_4h_1d_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(21) on 4h for trend
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    # Trend: 1 = rising (price > EMA), -1 = falling (price < EMA)
    trend_4h = np.zeros_like(close_4h)
    trend_4h[21:] = np.where(close_4h[21:] > ema_4h[21:], 1, 
                              np.where(close_4h[21:] < ema_4h[21:], -1, 0))
    # Align trend to 1h timeframe
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d for trend
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    # Trend: 1 = rising (price > EMA), -1 = falling (price < EMA)
    trend_1d = np.zeros_like(close_1d)
    trend_1d[50:] = np.where(close_1d[50:] > ema_1d[50:], 1, 
                              np.where(close_1d[50:] < ema_1d[50:], -1, 0))
    # Align trend to 1h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Session filter: 08-20 UTC (pre-compute for efficiency) ===
    # prices['open_time'] is already datetime64[ms]
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20, 21, 50)  # sufficient for Donchian, volume MA, EMAs
    
    for i in range(warmup, n):
        # --- Skip if outside trading session ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(trend_4h_aligned[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 12 bars (~12h on 1h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Require BOTH 4h and 1d trend alignment
            trend_long = (trend_4h_aligned[i] > 0) and (trend_1d_aligned[i] > 0)
            trend_short = (trend_4h_aligned[i] < 0) and (trend_1d_aligned[i] < 0)
            
            # Long: price breaks above Donchian upper AND both trends up
            if price > upper_20[i] and trend_long:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower AND both trends down
            elif price < lower_20[i] and trend_short:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals