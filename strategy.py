#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Supertrend trend filter with volume spike and ATR-based dynamic stoploss.
# Long when 1w Supertrend is bullish, volume > 2x 20-period average, and price > 1w Supertrend line.
# Short when 1w Supertrend is bearish, volume > 2x 20-period average, and price < 1w Supertrend line.
# Exit when price crosses the Supertrend line in the opposite direction or ATR-based trailing stop is hit.
# Uses discrete position size 0.30. Supertrend provides robust trend identification, volume confirms momentum.
# Target: 30-100 total trades over 4 years (7-25/year) with strong performance in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: Supertrend (ATR=10, mult=3.0) ===
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (10-period)
    atr_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_1w + low_1w) / 2.0
    upper_band = hl2 + (3.0 * atr_1w)
    lower_band = hl2 - (3.0 * atr_1w)
    
    # Supertrend calculation
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i-1] > upper_band[i-1]:
            direction[i] = 1
        elif close_1w[i-1] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align 1w Supertrend and direction to 1d timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Get 1d data for volume and ATR (for stoploss)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Volume moving average (20-period) on 1d
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # ATR (14-period) on 1d for dynamic stoploss
    tr1d = high_1d - low_1d
    tr2d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1d[0] = 0
    tr2d[0] = 0
    tr3d[0] = 0
    tr_d = np.maximum(tr1d, np.maximum(tr2d, tr3d))
    atr_14_1d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            stop_price = 0.0
            continue
        
        # Current values
        st_val = supertrend_aligned[i]
        dir_val = direction_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        atr_val = atr_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Update stoploss for existing position
        if position == 1:  # Long position
            # Trailing stop: highest high since entry minus 3*ATR
            if i == warmup or position == 0:  # New position or warmup
                entry_price = price
                stop_price = price - 3.0 * atr_val
            else:
                # Update highest high and trailing stop
                if price > entry_price:
                    entry_price = price  # Trail entry price up to current high for simplicity
                stop_price = max(stop_price, price - 3.0 * atr_val)
            
            # Exit if price breaks Supertrend line downward or hits stoploss
            if price < st_val or price <= stop_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                stop_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Trailing stop: lowest low since entry plus 3*ATR
            if i == warmup or position == 0:  # New position or warmup
                entry_price = price
                stop_price = price + 3.0 * atr_val
            else:
                # Update lowest low and trailing stop
                if price < entry_price:
                    entry_price = price  # Trail entry price down to current low for simplicity
                stop_price = min(stop_price, price + 3.0 * atr_val)
            
            # Exit if price breaks Supertrend line upward or hits stoploss
            if price > st_val or price >= stop_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                stop_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 2x 20-period average
            vol_filter = vol > 2.0 * vol_ma_val
            
            # LONG: Supertrend bullish, price above Supertrend line, volume spike
            if (dir_val > 0) and (price > st_val) and vol_filter:
                signals[i] = 0.30
                position = 1
                entry_price = price
                stop_price = price - 3.0 * atr_val
            
            # SHORT: Supertrend bearish, price below Supertrend line, volume spike
            elif (dir_val < 0) and (price < st_val) and vol_filter:
                signals[i] = -0.30
                position = -1
                entry_price = price
                stop_price = price + 3.0 * atr_val
        
        else:
            # Maintain current position size
            signals[i] = position * 0.30
    
    return signals

name = "1d_1wSupertrend_VolumeSpike_ATRTrailingStop_V1"
timeframe = "1d"
leverage = 1.0