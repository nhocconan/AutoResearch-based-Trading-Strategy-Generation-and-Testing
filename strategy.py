#!/usr/bin/env python3
"""
Experiment #095: 6h Volume-Weighted RSI + 1w Trend Filter

HYPOTHESIS: Volume-weighted RSI (VW-RSI) on 6h timeframe identifies overextended conditions,
while 1-week trend filter (price > EMA50) ensures alignment with higher timeframe direction.
VW-RSI gives more weight to price moves on high volume, reducing false signals from low-volume
noise. Strategy targets mean reversion at extreme VW-RSI levels (<20 for long, >80 for short)
with volume confirmation and trend filter. Designed to work in both bull and bear markets by
only taking trend-aligned entries. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vwrsi_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Calculate Volume-Weighted RSI(14)
    # VW-RSI = 100 - (100 / (1 + RS)), where RS = Avg Gain_Vol / Avg Loss_Vol
    # Gain_Vol = max(close - prev_close, 0) * volume
    # Loss_Vol = max(prev_close - close, 0) * volume
    
    price_change = np.diff(close, prepend=close[0])
    gain_vol = np.where(price_change > 0, price_change * volume, 0.0)
    loss_vol = np.where(price_change < 0, -price_change * volume, 0.0)
    
    # Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    avg_gain_vol = np.zeros(n)
    avg_loss_vol = np.zeros(n)
    
    # Initialize with simple average for first period
    if n >= period:
        avg_gain_vol[period-1] = np.mean(gain_vol[:period])
        avg_loss_vol[period-1] = np.mean(loss_vol[:period])
        
        # Wilder's smoothing for rest
        for i in range(period, n):
            avg_gain_vol[i] = alpha * gain_vol[i] + (1 - alpha) * avg_gain_vol[i-1]
            avg_loss_vol[i] = alpha * loss_vol[i] + (1 - alpha) * avg_loss_vol[i-1]
    
    # Avoid division by zero
    rs = np.zeros(n)
    vw_rsi = np.zeros(n)
    for i in range(n):
        if avg_loss_vol[i] == 0:
            rs[i] = np.inf
            vw_rsi[i] = 100.0
        elif avg_gain_vol[i] == 0:
            rs[i] = 0
            vw_rsi[i] = 0.0
        else:
            rs[i] = avg_gain_vol[i] / avg_loss_vol[i]
            vw_rsi[i] = 100.0 - (100.0 / (1.0 + rs[i]))
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, period * 2)  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1w trend ---
        price_above_1w_ema = close[i] > ema_50_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
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
                # Exit when VW-RSI returns to neutral territory (40-60)
                if 40 <= vw_rsi[i] <= 60:
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
                # Exit when VW-RSI returns to neutral territory (40-60)
                if 40 <= vw_rsi[i] <= 60:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: VW-RSI oversold (<20) in uptrend
        long_condition = (
            vw_rsi[i] < 20 and price_above_1w_ema
        )
        
        # Short: VW-RSI overbought (>80) in downtrend
        short_condition = (
            vw_rsi[i] > 80 and price_below_1w_ema
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