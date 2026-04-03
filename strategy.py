#!/usr/bin/env python3
"""
Experiment #794: 1h Strategy with 4h/1d HTF Filters
HYPOTHESIS: Use 4h Donchian(20) for trend direction and 1d volume spike for confirmation, 
enter on 1h pullbacks to VWAP. This captures momentum with precise timing while 
avoiding overtrading. Works in bull/bear: 4h Donchian filters trend, 1d volume 
confirms institutional interest. Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_794_1h_donchian20_1d_vol_vwap_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for Donchian trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian Channel(20) on 4h
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20_4h, lower_20_4h = donchian_channel(high_4h, low_4h, 20)
    # Trend: 1 = bullish (price above upper), -1 = bearish (price below lower), 0 = neutral
    donchian_trend_4h = np.zeros_like(upper_20_4h)
    # Need 4h close for comparison - get it
    close_4h = df_4h['close'].values
    donchian_trend_4h = np.where(close_4h > upper_20_4h, 1, 
                                 np.where(close_4h < lower_20_4h, -1, 0))
    # Align trend to 1h timeframe
    donchian_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_trend_4h)
    
    # === HTF: 1d data for volume spike filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate volume ratio (current / 20-day average) on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones_like(volume_1d)
    vol_ratio_1d[20:] = volume_1d[20:] / vol_ma_1d[20:]
    # Align volume ratio to 1h timeframe
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1h Indicators: VWAP for entry timing ===
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_volume = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.where(cum_volume > 0, cum_pv / cum_volume, 0.0)
    
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
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    warmup = max(20, 20)  # sufficient for HTF indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_trend_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(vwap[i]) or np.isnan(atr[i])):
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
            
            # Optional: time-based exit after 24 bars (~1 day on 1h) to avoid overtrading
            if bars_since_entry > 24:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average on 1d)
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        if volume_spike:
            # Long: 4h bullish trend AND price pulls back to VWAP (or below) on 1h
            if donchian_trend_4h_aligned[i] > 0 and price <= vwap[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: 4h bearish trend AND price pulls back to VWAP (or above) on 1h
            elif donchian_trend_4h_aligned[i] < 0 and price >= vwap[i]:
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