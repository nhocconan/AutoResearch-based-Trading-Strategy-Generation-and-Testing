#!/usr/bin/env python3
"""
Experiment #334: 1h HTF 4h/1d Donchian Breakout with Volume and Session Filter

HYPOTHESIS: Donchian channel breakouts on 4h and 1d timeframes provide strong directional signals.
Trading in the direction of HTF breakouts with volume confirmation on 1h timeframe captures
momentum moves while minimizing false breakouts. Session filter (08-20 UTC) avoids low-liquidity
periods. Using 1h only for entry timing reduces trade frequency to target 15-37/year.
Works in bull markets (breakouts with continuation) and bear markets (failed reversals at
HTF levels act as continuation signals in the dominant trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_334_1h_donchian_4h_1d_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for Donchian channel (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Donchian channel (20-period) for 4h
    def calculate_donchian(high_series, low_series, period=20):
        """Calculate Donchian upper and lower bands"""
        upper = pd.Series(high_series).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low_series).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_4h, lower_4h = calculate_donchian(df_4h['high'].values, df_4h['low'].values)
    
    # Align Donchian levels to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # === HTF: 1d data for Donchian channel (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channel (20-period) for 1d
    upper_1d, lower_1d = calculate_donchian(df_1d['high'].values, df_1d['low'].values)
    
    # Align Donchian levels to 1h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
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
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Session filter: 08-20 UTC ===
    # open_time is already datetime64[ms], access hour via index
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for HTF indicators stability
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            if in_position:
                # Continue holding position but don't allow new entries outside session
                signals[i] = position_side * SIZE
            else:
                signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or
            np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Determine HTF bias: 1 if bullish (price above both HTF Donchian middles), -1 if bearish
        # Calculate midpoints of HTF Donchian channels
        mid_4h = (upper_4h_aligned[i] + lower_4h_aligned[i]) / 2.0
        mid_1d = (upper_1d_aligned[i] + lower_1d_aligned[i]) / 2.0
        
        # HTF is bullish if price above both mids, bearish if below both mids
        htf_bullish = (price > mid_4h) and (price > mid_1d)
        htf_bearish = (price < mid_4h) and (price < mid_1d)
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        # Long entry: HTF bullish + price breaks above 4h upper Donchian + volume spike
        long_entry = htf_bullish and (price > upper_4h_aligned[i]) and volume_spike
        
        # Short entry: HTF bearish + price breaks below 4h lower Donchian + volume spike
        short_entry = htf_bearish and (price < lower_4h_aligned[i]) and volume_spike
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals