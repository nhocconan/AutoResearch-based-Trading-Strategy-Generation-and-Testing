#!/usr/bin/env python3
"""
Experiment #2320: 4h Donchian(20) Breakout + 1d EMA Trend + Volume Spike
HYPOTHESIS: Donchian channel breakouts on 4h capture strong momentum with 1d EMA trend filter and volume confirmation.
- Primary: 4h Donchian(20) breakout (long at 20-bar high, short at 20-bar low)
- HTF: 1d EMA(50) trend alignment (must agree for bias)
- Entry: Long when price breaks above Donchian(20) high with 1d uptrend + volume spike (>2.0x 20-bar average)
         Short when price breaks below Donchian(20) low with 1d downtrend + volume spike
- Exit: ATR(14) stoploss (2*ATR from entry) or opposite Donchian touch
- Volume: Require > 2.0x 20-bar average spike to confirm institutional participation
- Target: 75-200 total trades over 4 years (19-50/year) - suitable for 4h timeframe
- Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2320_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian(20), ATR(14), Volume MA(20) ===
    # Donchian(20): 20-bar high and low
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit if price moves 2*ATR against position
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches opposite Donchian band (lower band)
                elif price <= low_roll[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches opposite Donchian band (upper band)
                elif price >= high_roll[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 2.0x average - strict to limit trades)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike and trend_bias != 0:
            # Long entry: price breaks above Donchian(20) high with 1d uptrend
            if trend_bias > 0 and price >= high_roll[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian(20) low with 1d downtrend
            elif trend_bias < 0 and price <= low_roll[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals