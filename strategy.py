#!/usr/bin/env python3
"""
Experiment #3155: 6h Camarilla Pivot Reversal with 1w Trend Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for reversals, R4/S4 for breakouts) on 6h charts 
capture institutional order flow. 1-week EMA(50) trend filter ensures alignment with higher 
timeframe direction. Volume confirmation (>1.8x 20-period average) filters false signals. 
Position size 0.25. Target: 75-200 total trades over 4 years (19-50/year). 
Designed for ranging markets (reversals at R3/S3) and trending markets (breakouts at R4/S4).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3155_6h_camarilla1w_vol_v1"
timeframe = "6h"
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
    
    # Calculate EMA(50) on 1w close
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === 6h Indicators: Camarilla pivot levels from previous bar ===
    # Camarilla: based on previous bar's range
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # first bar has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    range_prev = prev_high - prev_low
    camarilla_multiplier = range_prev * 1.1 / 12  # 1.1/12 factor
    
    # Resistance levels
    r3 = prev_close + camarilla_multiplier * 3
    r4 = prev_close + camarilla_multiplier * 4
    # Support levels
    s3 = prev_close - camarilla_multiplier * 3
    s4 = prev_close - camarilla_multiplier * 4
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_loss = 0.0
    
    warmup = max(50, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Stoploss: 2.0 * ATR against position
            if position_side > 0:  # Long
                if price < stop_loss:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > stop_loss:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # 1w EMA trend filter
            price_vs_ema = price - ema_1w_aligned[i]
            
            # Long reversal at S3 (price rejects support with bullish 1w trend)
            if price <= s3[i] and price_vs_ema > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                stop_loss = entry_price - 2.0 * atr[i]
                signals[i] = SIZE
            # Short reversal at R3 (price rejects resistance with bearish 1w trend)
            elif price >= r3[i] and price_vs_ema < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                stop_loss = entry_price + 2.0 * atr[i]
                signals[i] = -SIZE
            # Long breakout above R4 (continuation with bullish 1w trend)
            elif price >= r4[i] and price_vs_ema > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                stop_loss = entry_price - 2.0 * atr[i]
                signals[i] = SIZE
            # Short breakdown below S4 (continuation with bearish 1w trend)
            elif price <= s4[i] and price_vs_ema < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                stop_loss = entry_price + 2.0 * atr[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals