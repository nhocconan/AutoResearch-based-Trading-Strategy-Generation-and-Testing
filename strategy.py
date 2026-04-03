#!/usr/bin/env python3
"""
Experiment #314: 1h Donchian(15) Breakout + 4h HMA Trend + Volume Spike

HYPOTHESIS: 1h Donchian channel breakouts filtered by 4h Hull Moving Average trend 
and volume spikes (>2.0x average) capture strong momentum moves with reduced false 
breakouts. The 4h HMA provides a medium-term trend filter, balancing responsiveness 
and stability. 1h timeframe targets 15-37 trades/year (60-150 total over 4 years) 
by using 4h for signal direction and 1h only for entry timing. Added session filter 
(08-20 UTC) to reduce noise trades. Uses ATR-based stoploss for risk management.

IMPLEMENTATION NOTES:
- Uses discrete position sizing (0.20) to minimize churn
- Volume confirmation threshold set to 2.0x to balance frequency and reliability
- Session filter: only trade 08:00-20:00 UTC
- Warmup period set to 100 bars to ensure stable indicators
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_314_1h_donchian_4h_hma_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for HMA trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HMA(21) on 4h data
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).rolling(window=half_period, min_periods=half_period).mean().values
        wma_full = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        raw_hma = 2.0 * wma_half - wma_full
        hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).mean().values
        return hma
    
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # === HTF: 1d data for regime filter (optional) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # === 1h Indicators: Donchian Channel (15) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(15, n):
        donchian_h[i] = np.max(high[i-15:i])
        donchian_l[i] = np.min(low[i-15:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Increased warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08:00-20:00 UTC ---
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- 4h HMA Trend Filter: Price > HMA = bullish bias, Price < HMA = bearish bias ---
        price_above_hma = close[i] > hma_4h_aligned[i]
        price_below_hma = close[i] < hma_4h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] < donchian_m[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] > donchian_m[i]:
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
        # Long: Donchian breakout up + volume spike + price above 4h HMA
        long_condition = breakout_up and volume_spike and price_above_hma
        
        # Short: Donchian breakout down + volume spike + price below 4h HMA
        short_condition = breakout_down and volume_spike and price_below_hma
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals