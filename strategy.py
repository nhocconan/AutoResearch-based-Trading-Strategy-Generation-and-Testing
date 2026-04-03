#!/usr/bin/env python3
"""
Experiment #376: 12h Donchian Breakout + 1d Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 12h timeframe, confirmed by 1d volume spikes (>2x average),
with ATR-based stoploss (2.5x) and discrete position sizing (0.25). Uses 1d trend filter (price > EMA50)
to avoid counter-trend trades. Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
to minimize fee drag while capturing high-probability breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike and EMA trend filter (Call ONCE before loop) ===
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
    
    # === 12h Indicators ===
    # Calculate Donchian channel (20-period) on 12h using HTF data
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) >= 20:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        
        # Donchian upper (20-period high) and lower (20-period low)
        donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
        
        # Align to 12h timeframe (already aligned as df_12h is 12h data)
        donch_high_aligned = donch_high  # No need to align as we're using 12h data directly
        donch_low_aligned = donch_low
        
        # But we need to map these values to the 12h bars in our primary timeframe
        # Since our primary timeframe IS 12h, we can use the values directly
        # However, we need to align the HTF 12h data to our 12h prices (which should be identical)
        # For safety, we'll use the align function
        donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
        donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    else:
        donch_high_12h_aligned = np.full(n, np.nan)
        donch_low_12h_aligned = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1d EMA50 ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using available data up to i
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
                # Exit if price breaks below Donchian lower (failed breakout)
                if close[i] < donch_low_12h_aligned[i]:
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
                # Exit if price breaks above Donchian upper (failed breakdown)
                if close[i] > donch_high_12h_aligned[i]:
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
            close[i] > donch_high_12h_aligned[i] and 
            volume_spike and 
            price_above_1d_ema
        )
        
        # Short: Price breaks below Donchian lower with volume confirmation and downtrend
        short_condition = (
            close[i] < donch_low_12h_aligned[i] and 
            volume_spike and 
            price_below_1d_ema
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