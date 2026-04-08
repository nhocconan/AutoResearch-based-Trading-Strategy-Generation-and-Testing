#!/usr/bin/env python3
"""
Experiment #6251: 6h Donchian(20) breakout + 1d Ichimoku cloud filter + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d Ichimoku cloud (bullish/bearish) capture institutional order flow with trend confirmation. Volume >2.0x average confirms participation. Discrete sizing (0.25) manages fee drag. Target: 75-150 trades over 4 years (19-37/year) for 6h timeframe.
Uses 1d Ichimoku for trend direction (proven effective in capturing sustained moves).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6251_6h_donchian20_1d_ichimoku_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for Ichimoku cloud ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 52:  # Need at least 52 periods for Ichimoku
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
        period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
        period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
        period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
        period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
        senkou_span_b = ((period52_high + period52_low) / 2)
        
        # Align to 6h timeframe (shift(1) inside align_htf_to_ltf for completed bars only)
        tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
        kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
        senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
        senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    else:
        tenkan_sen_aligned = np.full(n, np.nan)
        kijun_sen_aligned = np.full(n, np.nan)
        senkou_span_a_aligned = np.full(n, np.nan)
        senkou_span_b_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 52) + 1  # Donchian, volume avg, ATR, Ichimoku + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (21:00-23:59 UTC) ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                # OR price falls below Ichimoku cloud (trend reversal)
                cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
                if price <= stop_price or price <= donchian_low[i] or price < cloud_bottom:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                # OR price rises above Ichimoku cloud (trend reversal)
                cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
                if price >= stop_price or price >= donchian_high[i] or price > cloud_top:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0  # Volume filter for stronger signals
        
        # Ichimoku-based entry logic:
        # Long: breakout above Donchian high with volume AND bullish cloud (price above cloud)
        # Short: breakout below Donchian low with volume AND bearish cloud (price below cloud)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        bullish_cloud = price > cloud_top
        bearish_cloud = price < cloud_bottom
        
        long_entry = breakout_up and volume_confirmed and bullish_cloud
        short_entry = breakout_down and volume_confirmed and bearish_cloud
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals