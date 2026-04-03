#!/usr/bin/env python3
"""
Experiment #1159: 6h Camarilla Pivot + 12h Trend + Volume Confirmation
HYPOTHESIS: Camarilla pivot levels on 6h timeframe identify key support/resistance zones. 
Trend filter from 12h timeframe (EMA21 cross) prevents counter-trend entries. 
Volume confirmation (>1.3x average) ensures participation. Entries occur at R3/S3 (fade) 
and R4/S4 (breakout) with trend alignment. Designed for 6h timeframe with target 
50-150 total trades over 4 years (12-37/year). Works in both bull and bear markets 
by using trend-following breakouts and mean-reversion fades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1159_6h_camarilla_pivot_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # EMA21 trend: price > EMA21 = uptrend, < = downtrend
    ema_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === 6h Indicators: Camarilla Pivot Levels (based on previous bar) ===
    # Camarilla levels calculated from previous bar's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    r4 = pivot + (range_hl * 1.1 / 2.0)
    s4 = pivot - (range_hl * 1.1 / 2.0)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = 20  # sufficient for volume MA and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(r4[i]) or np.isnan(s4[i])):
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
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.3x average)
        volume_spike = vol_ratio[i] > 1.3
        
        if volume_spike:
            trend = trend_12h_aligned[i]
            
            # Fade at R3/S3 (mean reversion) - counter to trend
            if price >= r3[i] and trend < 0:  # 12h downtrend, fade R3 resistance
                in_position = True
                position_side = -1  # short
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            elif price <= s3[i] and trend > 0:  # 12h uptrend, fade S3 support
                in_position = True
                position_side = 1  # long
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Breakout continuation at R4/S4 (trend following) - with trend
            elif price >= r4[i] and trend > 0:  # 12h uptrend, break R4 resistance
                in_position = True
                position_side = 1  # long
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price <= s4[i] and trend < 0:  # 12h downtrend, break S4 support
                in_position = True
                position_side = -1  # short
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals