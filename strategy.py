#!/usr/bin/env python3
"""
Experiment #5277: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: On 4h timeframe, price breaking above/below Donchian(20) channel with HMA(21) trend confirmation and volume spike captures strong momentum moves. Volume confirmation reduces false breakouts. ATR-based stoploss manages risk. Designed for 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to minimize fee drag. Works in bull markets by catching upward breakouts and in bear markets by catching downward breakouts, while avoiding whipsaws in ranging markets via volume and trend filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5277_4h_donchian20_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for regime filter (optional, can add later) ===
    # For now, we'll use 1d EMA50 as regime filter if needed
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().shift(1).values
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Upper band: 20-period high
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle band: 20-period average (optional)
    donch_mid = (donch_high + donch_low) / 2
    
    # === 4h Indicators: HMA (21) for trend ===
    def hull_moving_average(arr, period):
        """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA helper
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        # Handle edge cases
        if period < 1:
            return np.full_like(arr, np.nan)
            
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        
        # 2*WMA(half) - WMA(full)
        diff = 2 * wma_half - wma_full
        
        # WMA of diff with sqrt_period
        hull = wma(diff, sqrt_period)
        
        # Align to original array length (accounting for shifts)
        result = np.full_like(arr, np.nan)
        # Calculate start index for hull values
        start_idx = period - half_period  # Approximate alignment
        end_idx = start_idx + len(hull)
        if end_idx <= len(arr) and start_idx >= 0:
            result[start_idx:end_idx] = hull
        return result
    
    hma_21 = hull_moving_average(close, 21)
    
    # === 4h Indicators: ATR (14) for stoploss ===
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev = np.roll(close, 1)
    close_prev[0] = np.nan
    tr = true_range(high, low, close_prev)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 4h Indicators: Volume spike (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)  # 50% above average
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0  # 1 for long, -1 for short
    entry_price = 0.0
    entry_atr = 0.0
    
    warmup = max(20, 21, 14, 20, 50)  # Donchian, HMA, ATR, volume, EMA50 warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (optional) ---
        hour = hours[i]
        # Trade during active hours: 00-24 UTC (can adjust if needed)
        # For now, trade all hours as 4h timeframe already filters
        
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(hma_21[i]) or np.isnan(atr_14[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR stoploss or trend reversal ---
        if in_position:
            # Calculate ATR-based stoploss levels
            if position_side > 0:  # Long position
                stop_loss = entry_price - 2.5 * entry_atr
                # Exit if price hits stoploss or trend turns bearish
                if price <= stop_loss or hma_21[i] < donch_mid[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                stop_loss = entry_price + 2.5 * entry_atr
                # Exit if price hits stoploss or trend turns bullish
                if price >= stop_loss or hma_21[i] > donch_mid[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donch_high[i-1]  # Price breaks above previous upper band
        breakout_down = price < donch_low[i-1]  # Price breaks below previous lower band
        
        # HMA trend confirmation
        hma_bullish = hma_21[i] > hma_21[i-1]  # HMA rising
        hma_bearish = hma_21[i] < hma_21[i-1]  # HMA falling
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Optional: regime filter (price vs 1d EMA50)
        regime_bullish = price > ema_50_aligned[i] if not np.isnan(ema_50_aligned[i]) else True
        regime_bearish = price < ema_50_aligned[i] if not np.isnan(ema_50_aligned[i]) else True
        
        # Entry conditions: Breakout + trend + volume + regime
        if breakout_up and hma_bullish and vol_confirm and regime_bullish:
            in_position = True
            position_side = 1
            entry_price = price
            entry_atr = atr_14[i]
            signals[i] = SIZE
        elif breakout_down and hma_bearish and vol_confirm and regime_bearish:
            in_position = True
            position_side = -1
            entry_price = price
            entry_atr = atr_14[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals