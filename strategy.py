#!/usr/bin/env python3
"""
Experiment #2534: 1h RSI(2) mean reversion with 4h trend filter and session filter
HYPOTHESIS: Intraday mean reversion on 1h timeframe works when aligned with 4h trend direction.
Uses RSI(2) for extreme short-term reversals (<10 for long, >90 for short) with volume confirmation.
Session filter (08-20 UTC) reduces noise during low-liquidity hours. 4h EMA(50) provides trend bias.
Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing (0.20) to minimize fee drag.
Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2534_1h_rsi2_4htrend_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for EMA trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(50)
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === 1h Indicators: RSI(2), Volume MA(20) ===
    # RSI(2) - very short term for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, min_periods=2, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, min_periods=2, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Session filter: 08-20 UTC ===
    # open_time is already datetime64[ms], use DatetimeIndex for .hour
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter ---
        hour = hours[i]
        if hour < 8 or hour > 20:  # Outside 08-20 UTC
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Mean reversion complete ---
        if in_position:
            if position_side > 0:  # Long - exit when RSI returns to neutral (50)
                if rsi[i] >= 50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short - exit when RSI returns to neutral (50)
                if rsi[i] <= 50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average)
        volume_confirmed = vol_ratio[i] > 1.5
        
        if volume_confirmed and trend_4h_aligned[i] != 0:
            # Long entry: RSI < 10 (extreme oversold) in uptrend
            if trend_4h_aligned[i] > 0 and rsi[i] < 10:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            # Short entry: RSI > 90 (extreme overbought) in downtrend
            elif trend_4h_aligned[i] < 0 and rsi[i] > 90:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals