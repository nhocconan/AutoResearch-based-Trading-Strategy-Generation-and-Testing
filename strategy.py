#!/usr/bin/env python3
"""
Experiment #788: 12h Donchian(20) breakout + 1w/1d HTF trend filter + Volume Spike
HYPOTHESIS: 12h Donchian breakouts capture medium-term momentum, filtered by 1w EMA trend 
direction and 1d volume confirmation (>1.8x average). Long when price breaks above 
Donchian upper AND 1w EMA rising AND volume spike. Short when price breaks below 
Donchian lower AND 1w EMA falling AND volume spike. Uses ATR(14) stoploss (2.0x). 
Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year). 
Works in bull/bear: in bull markets, 1w EMA rising filters for longs; in bear markets, 
1w EMA falling filters for shorts. Discrete position sizing (0.25) minimizes fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_788_12h_donchian20_1w_1d_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for EMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    # Trend: 1 = rising (ema > previous ema), -1 = falling (ema < previous ema), 0 = flat
    ema_trend_1w = np.zeros_like(ema_1w)
    ema_trend_1w[1:] = np.where(ema_1w[1:] > ema_1w[:-1], 1, 
                                np.where(ema_1w[1:] < ema_1w[:-1], -1, 0))
    # Align trend to 12h timeframe
    ema_trend_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_trend_1w)
    
    # === HTF: 1d data for volume average (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    # Volume MA(20) on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    # Align volume MA to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 12h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
    # === 12h Indicators: Volume ratio (current / 1d MA) ===
    vol_ratio = np.ones(n)
    # Avoid division by zero or NaN
    valid_vol_ma = vol_ma_1d_aligned > 0
    vol_ratio[valid_vol_ma] = volume[valid_vol_ma] / vol_ma_1d_aligned[valid_vol_ma]
    vol_ratio[~valid_vol_ma] = 1.0  # neutral when MA not available
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20, 50)  # sufficient for Donchian, volume MA, EMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(ema_trend_1w_aligned[i]) or np.isnan(vol_ratio[i]) or
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
            
            # Optional: time-based exit after 8 bars (~4 days on 12h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.8x 1d average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long: price breaks above Donchian upper AND 1w EMA rising
            if price > upper_20[i] and ema_trend_1w_aligned[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower AND 1w EMA falling
            elif price < lower_20[i] and ema_trend_1w_aligned[i] < 0:
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