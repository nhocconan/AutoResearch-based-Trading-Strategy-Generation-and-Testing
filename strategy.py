#!/usr/bin/env python3
"""
Experiment #1332: 12h Camarilla Pivot Reversal + 1d Trend + Volume Spike
HYPOTHESIS: Camarilla pivot levels (L3, L4, H3, H4) on 12h act as intraday support/resistance. 
In strong 1d trends, price often reverses from these levels. Volume spike (>2x) confirms participation. 
Works in bull markets (buy L3/L4 in uptrend) and bear markets (sell H3/H4 in downtrend). 
ATR stoploss manages risk. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1332_12h_camarilla_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # 1d trend: EMA(20) slope
    ema_1d = pd.Series(close_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_1d = np.zeros(len(close_1d))
    trend_1d[20:] = np.where(ema_1d[20:] > ema_1d[:-20], 1, -1)  # rising/falling
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 12h Indicators: Camarilla Pivot Levels from previous day ===
    # Need previous day's high, low, close
    # Since we're on 12h, we approximate using rolling window of 2 bars (24h)
    # But better: use actual 1d data from HTF
    if len(df_1d) >= 2:
        prev_high = df_1d['high'].shift(1).values  # previous day's high
        prev_low = df_1d['low'].shift(1).values    # previous day's low
        prev_close = df_1d['close'].shift(1).values # previous day's close
    else:
        prev_high = df_1d['high'].values
        prev_low = df_1d['low'].values
        prev_close = df_1d['close'].values
    
    # Camarilla levels
    range_ = prev_high - prev_low
    camarilla_h3 = prev_close + range_ * 1.1 / 4
    camarilla_h4 = prev_close + range_ * 1.1 / 2
    camarilla_l3 = prev_close - range_ * 1.1 / 4
    camarilla_l4 = prev_close - range_ * 1.1 / 2
    
    # Align to 12h timeframe (shifted by 1 for completed day)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
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
    
    warmup = 20  # sufficient for EMA and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
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
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Reversal from Camarilla levels in direction of 1d trend
            if trend_1d_aligned[i] > 0:  # 1d uptrend
                # Buy near support (L3, L4)
                if price <= l3_12h[i] * 1.002 or price <= l4_12h[i] * 1.002:  # small buffer
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            else:  # 1d downtrend
                # Sell near resistance (H3, H4)
                if price >= h3_12h[i] * 0.998 or price >= h4_12h[i] * 0.998:  # small buffer
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
        
        # Default: no signal
        if not in_position:
            signals[i] = 0.0
    
    return signals