#!/usr/bin/env python3
"""
Experiment #2864: 1d Donchian(20) Breakout + 1w Trend Filter + Volume Confirmation
HYPOTHESIS: Daily Donchian(20) breakouts capture medium-term trends. Weekly trend filter ensures
we only trade in the direction of the higher timeframe trend (buy breakouts in weekly uptrend,
sell breakdowns in weekly downtrend), reducing false signals during counter-trend moves.
Volume confirmation adds conviction. Daily timeframe targets 30-100 trades over 4 years (7-25/year)
to minimize fee drag. ATR-based stoploss manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2864_1d_donchian20_1w_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Donchian channels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on daily data
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_roll_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_upper = high_roll_max
    donch_lower = low_roll_min
    
    # Align to 1d timeframe (prices is already 1d, so no shift needed for alignment)
    # But we still use align_htf_to_ltf for consistency and proper handling of gaps
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === ATR(14) for stoploss ===
    # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for all indicators (max of 20, 50, 14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Long exit: stoploss or reversal signal
            if position_side > 0:
                # Stoploss: 2 * ATR below entry
                if price <= entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit on weekly trend reversal
                elif trend_1w_aligned[i] < 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            # Short exit: stoploss or reversal signal
            else:
                # Stoploss: 2 * ATR above entry
                if price >= entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit on weekly trend reversal
                elif trend_1w_aligned[i] > 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Get weekly trend bias
            trend_bias = trend_1w_aligned[i]
            
            # Long entry: price breaks above Donchian upper in weekly uptrend
            if trend_bias > 0 and price > donch_upper_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower in weekly downtrend
            elif trend_bias < 0 and price < donch_lower_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals