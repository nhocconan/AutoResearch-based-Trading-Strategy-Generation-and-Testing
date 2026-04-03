#!/usr/bin/env python3
"""
Experiment #272: 12h Donchian Breakout + 1d EMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Combining 12h Donchian(20) breakouts with 1d EMA(50) trend alignment and 
volume confirmation creates a robust trend-following strategy. The 12h timeframe minimizes 
fee drag while capturing medium-term trends. Volume spike confirms institutional participation 
in breakouts. EMA(50) on 1d ensures we only trade in the direction of the higher timeframe 
trend, reducing whipsaws. ATR-based stoploss manages risk. Targets 12-37 trades/year on 
12h timeframe (50-150 total over 4 years) to minimize fee drag while maintaining statistical 
significance. Designed to work in both bull (breakouts) and bear (breakdowns) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for regime filter (optional, using 1d as proxy) ===
    # We'll use 1d volatility as regime filter instead
    
    # === 12h Indicators ===
    # Donchian Channels (20-period)
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # Volume Spike Detection (Volume > 2.0 * 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR(14) for stoploss and volatility filter
    def atr(high, low, close, period):
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]  # First TR
        atr_vals = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        return atr_vals
    
    atr_14 = atr(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or np.isnan(volume_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks below Donchian lower (20) or reverses against trend
                if close[i] < lower_20[i] or close[i] < ema_50_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks above Donchian upper (20) or reverses against trend
                if close[i] > upper_20[i] or close[i] > ema_50_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper (20) with volume spike and above 1d EMA
        if (close[i] > upper_20[i] and 
            volume_spike[i] and 
            close[i] > ema_50_1d_aligned[i]):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Price breaks below Donchian lower (20) with volume spike and below 1d EMA
        elif (close[i] < lower_20[i] and 
              volume_spike[i] and 
              close[i] < ema_50_1d_aligned[i]):
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals