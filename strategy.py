#!/usr/bin/env python3
"""
Experiment #4814: 1h Volume Spike + 4h/1d Trend Filter with Session
HYPOTHESIS: On 1h timeframe, volume spikes (>2x 20-bar average) in direction of 4h HMA21 and 1d close>open trend capture momentum bursts. Uses 4h/1d for signal direction (reducing whipsaw) and 1h only for entry timing. Session filter (08-20 UTC) avoids low-liquidity hours. Target: 60-150 trades over 4 years (15-37/year) to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (mean reversion off extremes with volume).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4814_1h_volume_spike_4h_1d_trend_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours (08-20 UTC) ONCE before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Precompute HTF: 4h data for HMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    # Precompute HTF: 1d data for daily trend filter (close > open)
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h Indicators: HMA21 for trend filter ===
    if len(df_4h) >= 21:
        # Hull Moving Average calculation
        half_len = len(df_4h) // 2
        sqrt_len = int(np.sqrt(len(df_4h)))
        
        # WMA function using convolution for speed
        def wma(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        close_4h = df_4h['close'].values
        wma_half = np.array([wma(close_4h[i:i+half_len], half_len)[-1] 
                            if i+half_len <= len(close_4h) else np.nan 
                            for i in range(len(close_4h))])
        wma_full = np.array([wma(close_4h[i:i+len(close_4h)], len(close_4h))[-1] 
                            if i+len(close_4h) <= len(close_4h) else np.nan 
                            for i in range(len(close_4h))])
        wma_sqrt = np.array([wma(close_4h[i:i+sqrt_len], sqrt_len)[-1] 
                            if i+sqrt_len <= len(close_4h) else np.nan 
                            for i in range(len(close_4h))])
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        hma_raw = 2 * wma_half - wma_full
        hma_4h = np.array([wma(hma_raw[i:i+sqrt_len], sqrt_len)[-1] 
                          if i+sqrt_len <= len(hma_raw) else np.nan 
                          for i in range(len(hma_raw))])
    else:
        hma_4h = np.full(len(df_4h), np.nan)
    
    # Align HTF HMA21 to 1h timeframe
    if len(hma_4h) > 0:
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    else:
        hma_4h_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Daily trend (close > open) ===
    if len(df_1d) >= 1:
        daily_bull = df_1d['close'].values > df_1d['open'].values  # True if bullish day
    else:
        daily_bull = np.array([])
    
    # Align HTF daily trend to 1h timeframe
    if len(daily_bull) > 0:
        daily_bull_aligned = align_htf_to_ltf(prices, df_1d, daily_bull.astype(float))
        # Convert back to boolean (alignment may produce floats)
        daily_bull_aligned = daily_bull_aligned > 0.5
    else:
        daily_bull_aligned = np.full(n, False)
    
    # === 1h Indicators: Volume confirmation (2x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20)  # Volume MA, HTF alignment warmup
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(daily_bull_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on opposite signal ---
        if in_position:
            # Exit conditions: volume spike in opposite direction OR daily trend flip
            vol_confirm = vol_ratio[i] > 2.0
            
            if position_side > 0:  # Long position
                # Exit if: volume spike short AND daily trend turns bearish
                exit_signal = vol_confirm and (price < hma_4h_aligned[i]) and (~daily_bull_aligned[i])
                if exit_signal:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit if: volume spike long AND daily trend turns bullish
                exit_signal = vol_confirm and (price > hma_4h_aligned[i]) and daily_bull_aligned[i]
                if exit_signal:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Entry conditions: volume spike + 4h trend alignment + daily trend filter
        entry_long = vol_confirm and (price > hma_4h_aligned[i]) and daily_bull_aligned[i]
        entry_short = vol_confirm and (price < hma_4h_aligned[i]) and (~daily_bull_aligned[i])
        
        # Final entry conditions
        if entry_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif entry_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals