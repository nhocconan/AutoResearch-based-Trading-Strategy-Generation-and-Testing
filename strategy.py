#!/usr/bin/env python3
"""
Experiment #834: 1h Donchian(20) + 4h/1d Trend Filter + Volume Spike + Session Filter
HYPOTHESIS: 1h Donchian breakouts capture momentum, filtered by 4h HMA and 1d EMA200 trend alignment 
and volume confirmation (>2.0x average) during active session (08-20 UTC). 
Long when price breaks above Donchian upper AND 4h HMA rising AND 1d close>EMA200 AND volume spike. 
Short when price breaks below Donchian lower AND 4h HMA falling AND 1d close<EMA200 AND volume spike. 
Uses discrete position sizing (0.20) to limit drawdown. Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_834_1h_donchian20_4h_hma_1d_ema_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 4h data for HMA trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate HMA(21) on 4h
    def calculate_hma(arr, period):
        half = int(period / 2)
        sqrt = int(np.sqrt(period))
        wma1 = pd.Series(arr).ewm(span=half, min_periods=half, adjust=False).mean().values
        wma2 = pd.Series(arr).ewm(span=period, min_periods=period, adjust=False).mean().values
        raw_hma = 2 * wma1 - wma2
        hma = pd.Series(raw_hma).ewm(span=sqrt, min_periods=sqrt, adjust=False).mean().values
        return hma
    
    hma_4h = calculate_hma(close_4h, 21)
    # Trend: 1 = rising (hma > previous hma), -1 = falling (hma < previous hma), 0 = flat
    hma_trend_4h = np.zeros_like(hma_4h)
    hma_trend_4h[1:] = np.where(hma_4h[1:] > hma_4h[:-1], 1, 
                                 np.where(hma_4h[1:] < hma_4h[:-1], -1, 0))
    # Align trend to 1h timeframe
    hma_trend_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_trend_4h)
    
    # === HTF: 1d data for EMA200 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(200) on 1d
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    # Trend: 1 = above EMA200, -1 = below EMA200
    ema_trend_1d = np.where(close_1d > ema_200_1d, 1, -1)
    # Align trend to 1h timeframe
    ema_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_1d)
    
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
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20, 200)  # sufficient for Donchian, volume MA, EMA200
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(hma_trend_4h_aligned[i]) or
            np.isnan(ema_trend_1d_aligned[i]) or np.isnan(atr[i])):
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
            # Long: price breaks above Donchian upper AND 4h HMA rising AND 1d above EMA200
            if price > upper_20[i] and hma_trend_4h_aligned[i] > 0 and ema_trend_1d_aligned[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower AND 4h HMA falling AND 1d below EMA200
            elif price < lower_20[i] and hma_trend_4h_aligned[i] < 0 and ema_trend_1d_aligned[i] < 0:
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