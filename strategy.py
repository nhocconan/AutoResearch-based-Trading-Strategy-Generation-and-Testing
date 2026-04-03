#!/usr/bin/env python3
"""
Experiment #355: 6h Elder Ray + 1d ADX Regime Filter + Volume Spike

HYPOTHESIS: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure,
filtered by 1d ADX regime (ADX>25 = trending, ADX<20 = ranging) and 6h volume spikes.
In trending regimes, we trade with the Elder Ray signal; in ranging regimes, we fade
extremes. This adapts to both bull/bear markets by shifting regime definitions. Target:
50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_1d_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX regime and EMA13 (for Elder Ray) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 30:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # ADX calculation (14-period)
        plus_dm = np.zeros(len(high_1d))
        minus_dm = np.zeros(len(high_1d))
        tr = np.zeros(len(high_1d))
        for i in range(1, len(high_1d)):
            plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
            minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
            tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                result[period-1] = np.nansum(data[:period])
                for i in range(period, len(data)):
                    result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        atr_1d = wilder_smooth(tr, 14)
        plus_di_1d = 100 * wilder_smooth(plus_dm, 14) / atr_1d
        minus_di_1d = 100 * wilder_smooth(minus_dm, 14) / atr_1d
        dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
        adx_1d = wilder_smooth(dx_1d, 14)
        
        # EMA13 for Elder Ray (using close)
        ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False).mean().values
        
        # Align to 6h timeframe
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
        ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    else:
        adx_1d_aligned = np.full(n, 20.0)  # Default to ranging
        ema13_1d_aligned = np.full(n, close.mean())
    
    # === 6h Indicators: Elder Ray Components ===
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    # Volume spike detection (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Definition ---
        is_trending = adx_1d_aligned[i] > 25
        is_ranging = adx_1d_aligned[i] < 20
        
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
                # Take profit at 2R or bear power exhaustion
                if close[i] >= entry_price + 5.0 * atr_14 or bear_power[i] > -0.1 * np.std(bull_power[max(0, i-50):i+1]):
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
                # Take profit at 2R or bull power exhaustion
                if close[i] <= entry_price - 5.0 * atr_14 or bull_power[i] < 0.1 * np.std(bull_power[max(0, i-50):i+1]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        vol_spike = vol_ratio[i] > 2.0
        
        if is_trending and vol_spike:
            # Trending regime: trade with Elder Ray signal
            if bull_power[i] > 0 and bear_power[i] < 0:  # Both confirm bullish
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif bear_power[i] < 0 and bull_power[i] < 0:  # Bear power negative, bull power weak
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
        elif is_ranging and vol_spike:
            # Ranging regime: fade extremes
            bull_std = np.std(bull_power[max(0, i-100):i+1]) if i >= 100 else np.std(bull_power[:i+1])
            bear_std = np.std(bear_power[max(0, i-100):i+1]) if i >= 100 else np.std(bear_power[:i+1])
            
            if bull_power[i] > 2.0 * bull_std and bull_power[i] > 0:  # Overbought
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            elif bear_power[i] < -2.0 * bear_std and bear_power[i] < 0:  # Oversold
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
        else:
            signals[i] = 0.0
    
    return signals