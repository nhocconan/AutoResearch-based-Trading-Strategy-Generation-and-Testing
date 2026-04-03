#!/usr/bin/env python3
"""
Experiment #406: 4h Donchian Breakout + 1d Volume Spike + 1d Trend Filter

HYPOTHESIS: 4h Donchian(20) breakouts with 1d volume confirmation (>1.8x average) and 
1d trend filter (price > EMA50 on daily) captures strong momentum moves in both bull 
(bullish breakouts) and bear (bearish breakdowns) markets. Using 4h primary timeframe 
with 1d HTF filters reduces noise and overtrading vs lower timeframes. Target: 75-200 
total trades over 4 years (19-50/year) to minimize fee drag while maintaining statistical 
significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike and trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # Calculate EMA(50) on 1d close for trend filter
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Calculate Donchian channels (20-period) ===
    if n >= 20:
        # Calculate rolling max/min for Donchian channels
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
        donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
        # For warmup period, fill with NaN
        donchian_upper[:19] = np.nan
        donchian_lower[:19] = np.nan
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
    
    # === Session filter: 00-23 UTC (trade all hours for 4h timeframe) ===
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Trade all hours for 4h timeframe ---
        hour = hours[i]
        # No session filter for 4h - trade continuously
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian lower (trailing stop for longs)
                if close[i] <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian upper (trailing stop for shorts)
                if close[i] >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper with volume confirmation and uptrend
        long_condition = (
            close[i] > donchian_upper[i] and  # Breakout above upper channel
            vol_ratio_1d_aligned[i] > 1.8 and  # Volume spike confirmation
            close[i] > ema_50_1d_aligned[i]   # Price above daily EMA50 (uptrend)
        )
        
        # Short: Price breaks below Donchian lower with volume confirmation and downtrend
        short_condition = (
            close[i] < donchian_lower[i] and  # Breakdown below lower channel
            vol_ratio_1d_aligned[i] > 1.8 and  # Volume spike confirmation
            close[i] < ema_50_1d_aligned[i]   # Price below daily EMA50 (downtrend)
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals