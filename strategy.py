#!/usr/bin/env python3
"""
Experiment #874: 1h Donchian(20) + 4h EMA Trend + 1d Volume Spike + Session Filter
HYPOTHESIS: Donchian breakouts on 1h capture momentum, filtered by 4h EMA trend direction 
and 1d volume confirmation (>1.8x average) during active UTC session (08-20). 
Long when price breaks above Donchian upper AND 4h EMA rising AND volume spike. 
Short when price breaks below Donchian lower AND 4h EMA falling AND volume spike. 
Uses discrete position sizing (0.20) to limit drawdown. Target: 60-150 total trades 
over 4 years (15-37/year) for 1h timeframe. Includes ATR-based stoploss (2.0) to 
control risk and prevent large losses in bear markets like 2022.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_874_1h_donchian20_4h_ema_1d_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 4h data for EMA trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(21) on 4h
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    # Trend: 1 = rising (ema > previous ema), -1 = falling (ema < previous ema), 0 = flat
    ema_trend_4h = np.zeros_like(ema_4h)
    ema_trend_4h[1:] = np.where(ema_4h[1:] > ema_4h[:-1], 1, 
                                 np.where(ema_4h[1:] < ema_4h[:-1], -1, 0))
    # Align trend to 1h timeframe
    ema_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_trend_4h)
    
    # === HTF: 1d data for volume spike filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate volume MA(20) on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones_like(volume_1d)
    vol_ratio_1d[20:] = volume_1d[20:] / vol_ma_1d[20:]
    # Align volume ratio to 1h timeframe
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20, 20)  # sufficient for Donchian, EMA, volume MA
    
    for i in range(warmup, n):
        # --- Skip if outside trading session ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(ema_trend_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr[i])):
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
            
            # Optional: time-based exit after 6 bars (~6h on 1h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        if volume_spike:
            # Long: price breaks above Donchian upper AND 4h EMA rising
            if price > upper_20[i] and ema_trend_4h_aligned[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower AND 4h EMA falling
            elif price < lower_20[i] and ema_trend_4h_aligned[i] < 0:
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