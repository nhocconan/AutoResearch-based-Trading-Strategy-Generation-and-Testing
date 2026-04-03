#!/usr/bin/env python3
"""
Experiment #1955: 6h Donchian(20) Breakout + 1w Trend Filter + Volume Spike
HYPOTHESIS: 6h Donchian breakouts aligned with weekly trend and volume confirmation capture institutional flow.
Weekly trend filter (price vs EMA50) ensures we trade with the higher-timeframe momentum.
Volume spike (>2x 20-period average) confirms participation. Target: 75-150 total trades over 4 years.
Works in bull/bear by following weekly institutional bias. Uses discrete position sizing (0.25) to minimize churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1955_6h_donchian20_1w_trend_vol_v1"
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
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_50_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 6h Indicators: Donchian(20) channels and Volume MA(20) ===
    # Donchian channels: upper = max(high, 20), lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume MA(20) for spike detection
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
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20), EMA(50), volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss and mean reversion exit ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for dynamic stoploss
            if i >= 14:
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                # Simple ATR calculation using rolling mean
                atr = np.mean([tr] + [abs(high[j] - low[j]) for j in range(max(0, i-13), i)])
            else:
                atr = price * 0.02  # fallback 2%
            
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2*ATR below entry
                if price < entry_price - 2.0 * atr:
                    exit_signal = True
                # Exit if price returns to Donchian lower (mean reversion)
                elif price <= donchian_lower[i]:
                    exit_signal = True
            else:  # Short position
                # Stoploss: 2*ATR above entry
                if price > entry_price + 2.0 * atr:
                    exit_signal = True
                # Exit if price returns to Donchian upper (mean reversion)
                elif price >= donchian_upper[i]:
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
        # Require weekly trend alignment
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require significant volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND weekly trend up
            if trend_bias > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND weekly trend down
            elif trend_bias < 0 and price < donchian_lower[i]:
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