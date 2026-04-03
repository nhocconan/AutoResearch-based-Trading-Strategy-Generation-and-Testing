#!/usr/bin/env python3
"""
Experiment #1929: 4h Donchian(20) Breakout + 1d Trend Filter + Volume Spike + ATR Stoploss
HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture institutional order flow when aligned with 1d trend and confirmed by volume spikes. 
Strategy: 
- Use 4h Donchian(20) for breakout signals (long: break above upper band, short: break below lower band)
- Require 1d EMA(50) trend filter to align with higher timeframe direction
- Volume confirmation: current volume > 2.0x 20-period average to filter weak breakouts
- ATR-based stoploss: exit when price moves 2.5*ATR against position
- Position size: 0.25 (25% of capital) discrete levels to minimize fee churn
- Works in bull/bear markets by following 1d institutional trend while capturing 4h momentum
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1929_4h_donchian20_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_hi = high_series.rolling(window=20, min_periods=20).max().values
    donch_lo = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20), EMA(50), ATR(14), volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions: ATR-based stoploss
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price drops 2.5*ATR below entry
                if price < entry_price - 2.5 * atr[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price rises 2.5*ATR above entry
                if price > entry_price + 2.5 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper band AND 1d trend up
            if trend_bias > 0 and price > donch_hi[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower band AND 1d trend down
            elif trend_bias < 0 and price < donch_lo[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals