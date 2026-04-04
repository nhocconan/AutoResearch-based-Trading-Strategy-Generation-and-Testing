#!/usr/bin/env python3
"""
Experiment #5373: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation
HYPOTHESIS: On 4h timeframe, price breaking above/below the 20-period Donchian channel 
with volume > 2.0x average and aligned with 12h Hull Moving Average trend captures 
strong momentum moves while minimizing overtrading. The 12h HMA provides a smoother 
trend filter than EMA/SMA, reducing whipsaws in ranging markets. Discrete position 
sizing (0.25) and ATR-based stoploss (2.0x ATR) control risk. Target: 19-50 trades/year 
(75-200 total over 4 years) to minimize fee drag while maintaining statistical significance. 
Works in bull markets via breakouts above rising 12h HMA and in bear markets via short 
breakdowns below falling 12h HMA.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5373_4h_donchian20_12h_hma_vol_v1"
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
    
    # === HTF: 12h data for HMA trend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 21:
        # Calculate HMA(21) on 12h close prices
        # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        close_12h = df_12h['close'].values
        if len(close_12h) >= 21:
            wma_full = np.convolve(close_12h, np.arange(1, 22), mode='valid') / (21 * 22 / 2)
            wma_half = np.convolve(close_12h, np.arange(1, half_len + 1), mode='valid') / (half_len * (half_len + 1) / 2)
            hma_12h = 2 * wma_half - wma_full
            hma_12h = np.convolve(hma_12h, np.arange(1, sqrt_len + 1), mode='valid') / (sqrt_len * (sqrt_len + 1) / 2)
            # Pad to match original length
            hma_12h_padded = np.full(len(close_12h), np.nan)
            hma_12h_padded[half_len + sqrt_len - 1:] = hma_12h
            hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_padded)
        else:
            hma_12h_aligned = np.full(n, np.nan)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20, 14, 21)  # Donchian, volume avg, ATR, HMA warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        # Trade during major sessions: 00-06 UTC (Asia), 07-12 UTC (Europe), 13-20 UTC (US)
        # Avoid 21-23 UTC (low liquidity between sessions)
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss ---
        if in_position:
            if position_side > 0:  # Long position
                # Stoploss: 2.0 * ATR below entry price
                stop_price = entry_price - 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. 12h HMA turns down (trend reversal)
                if price <= stop_price or price <= donchian_low[i] or hma_12h_aligned[i] < hma_12h_aligned[i-1]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Stoploss: 2.0 * ATR above entry price
                stop_price = entry_price + 2.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. 12h HMA turns up (trend reversal)
                if price >= stop_price or price >= donchian_high[i] or hma_12h_aligned[i] > hma_12h_aligned[i-1]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i-1]  # Break above previous period's high
        breakout_down = price < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = volume_ratio[i] > 2.0
        
        # 12h HMA trend filter
        # Long: 12h HMA rising (current > previous)
        # Short: 12h HMA falling (current < previous)
        hma_rising = hma_12h_aligned[i] > hma_12h_aligned[i-1]
        hma_falling = hma_12h_aligned[i] < hma_12h_aligned[i-1]
        
        # Entry conditions
        if breakout_up and volume_confirmed and hma_rising:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and hma_falling:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals