#!/usr/bin/env python3
"""
Experiment #333: 4h Donchian Breakout + 12h HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian(20) breakout on 4h timeframe, confirmed by 12h HMA(21) trend direction 
and volume spike (>1.5x 20-period average), creates a robust trend-following strategy 
that works in both bull and bear markets. The Donchian structure captures breakouts 
with clear risk definition, 12h HMA filters for higher timeframe trend alignment, 
and volume confirmation ensures institutional participation. Targets 19-50 trades/year 
on 4h timeframe (75-200 total over 4 years) to minimize fee drag while capturing 
significant market moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_12h_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close
    if len(df_12h) >= 21:
        close_12h = df_12h['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        wma_half = np.array([np.nan] * len(close_12h))
        wma_full = np.array([np.nan] * len(close_12h))
        
        if len(close_12h) >= half_len:
            wma_half[half_len-1:] = wma(close_12h, half_len)
        if len(close_12h) >= 21:
            wma_full[20:] = wma(close_12h, 21)
        
        hma_12h = np.array([np.nan] * len(close_12h))
        if len(close_12h) >= 21:
            diff = 2 * wma_half - wma_full
            valid_start = 20 + half_len - 1
            if len(diff) >= valid_start:
                hma_12h[valid_start:] = wma(diff[20:], sqrt_len)
        
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === HTF: 12h data for volume spike confirmation (Call ONCE before loop) ===
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === 4h Indicators ===
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high[19:] = high_series.rolling(window=20, min_periods=20).max().values[19:]
        donchian_low[19:] = low_series.rolling(window=20, min_periods=20).min().values[19:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss or opposite signal) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                # Exit conditions: stoploss hit OR Donchian low break (reversal signal)
                if low[i] < stop_level or close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                # Exit conditions: stoploss hit OR Donchian high break (reversal signal)
                if high[i] > stop_level or close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio_12h_aligned[i] > 1.5
        
        # Long: Price breaks above Donchian high + 12h HMA rising (trending up) + volume
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above upper band
            hma_12h_aligned[i] > hma_12h_aligned[i-1] and  # 12h HMA rising
            volume_spike
        )
        
        # Short: Price breaks below Donchian low + 12h HMA falling (trending down) + volume
        short_condition = (
            close[i] < donchian_low[i] and  # Breakdown below lower band
            hma_12h_aligned[i] < hma_12h_aligned[i-1] and  # 12h HMA falling
            volume_spike
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals