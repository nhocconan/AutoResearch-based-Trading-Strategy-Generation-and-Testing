#!/usr/bin/env python3
"""
Experiment #237: 4h Donchian Breakout + HMA Trend + Volume Confirmation + ATR Stoploss

HYPOTHESIS: Donchian(20) breakout captures momentum bursts, confirmed by 1d HMA(21) trend filter and volume spike (>1.8x MA20). 
In trending markets (1d HMA up/down), we follow breakouts. In ranging markets (1d HMA flat), we require stronger volume confirmation (>2.5x) to avoid false breakouts.
This structure works in both bull (clear breakouts) and bear (failed reversals, volatility expansion) markets. 
4h timeframe targets 19-50 trades/year (75-200 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_237_4h_donchian_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA(21) for trend filter
    def calculate_hma(series, period):
        """Hull Moving Average"""
        if len(series) < period:
            return np.full_like(series, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma1 = pd.Series(series).ewm(span=half, adjust=False).mean()
        wma2 = pd.Series(series).ewm(span=period, adjust=False).mean()
        raw_hma = 2 * wma1 - wma2
        hma = pd.Series(raw_hma).ewm(span=sqrt, adjust=False).mean()
        return hma.values
    
    hma_21_1d = calculate_hma(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 1d HMA slope for trend direction (up/down/flat)
    hma_slope_1d = np.zeros(n)
    hma_slope_1d[1:] = hma_21_1d_aligned[1:] - hma_21_1d_aligned[:-1]
    hma_slope_1d[0] = 0
    
    # === 4h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_20_upper, donch_20_lower = calculate_donchian(high, low, 20)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for 1d indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_1d_aligned[i]) or np.isnan(hma_slope_1d[i]) or
            np.isnan(donch_20_upper[i]) or np.isnan(donch_20_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d Regime Filter: HMA slope > 0 = up, < 0 = down, |slope| < threshold = flat ---
        hma_up = hma_slope_1d[i] > 0.0001 * close[i]  # Scaled threshold
        hma_down = hma_slope_1d[i] < -0.0001 * close[i]
        hma_flat = np.abs(hma_slope_1d[i]) <= 0.0001 * close[i]
        
        # --- Volume Confirmation: Dynamic based on regime ---
        if hma_flat:
            volume_spike = vol_ratio[i] > 2.5  # Stronger confirmation in ranging markets
        else:
            volume_spike = vol_ratio[i] > 1.8  # Normal confirmation in trending markets
        
        # --- Price ---
        price = close[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle reversion in flat markets
                if hma_flat and price < (donch_20_upper[i] + donch_20_lower[i]) / 2:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle reversion in flat markets
                if hma_flat and price > (donch_20_upper[i] + donch_20_lower[i]) / 2:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trending market logic: Follow Donchian breakout with trend
        if hma_up or hma_down:
            # Long: Price breaks above Donchian upper AND HMA up
            long_breakout = (price > donch_20_upper[i]) and hma_up
            
            # Short: Price breaks below Donchian lower AND HMA down
            short_breakout = (price < donch_20_lower[i]) and hma_down
            
            # Require volume confirmation for breakout entries
            if long_breakout and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_breakout and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        
        # Ranging market logic: Fade Donchian touches with strong volume
        else:  # hma_flat
            # Long: Price touches Donchian lower AND strong volume spike (mean reversion)
            long_reversion = (price <= donch_20_lower[i] * 1.001) and volume_spike  # Allow small slippage
            
            # Short: Price touches Donchian upper AND strong volume spike (mean reversion)
            short_reversion = (price >= donch_20_upper[i] * 0.999) and volume_spike  # Allow small slippage
            
            if long_reversion:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_reversion:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals