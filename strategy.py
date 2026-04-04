#!/usr/bin/env python3
"""
Experiment #5270: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation
HYPOTHESIS: On 1d timeframe, Donchian(20) breakouts capture strong momentum moves. The 1w HMA(21) filter ensures we only trade in the direction of the higher timeframe trend, avoiding counter-trend whipsaws. Volume confirmation (>1.5x average volume) adds conviction to breakouts. This combination works in both bull and bear markets by following the established trend on the 1w timeframe. Designed for 7-25 trades/year on 1d timeframe (30-100 total over 4 years) to minimize fee drag. Uses discrete position sizing (0.30) and ATR-based stoploss (signal→0 when price moves 2.5*ATR against position).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5270_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1w data for HMA trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Calculate HMA(21) on weekly close
        close_1w = df_1w['close'].values
        n_1w = len(close_1w)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        # WMA function
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        wma_half = np.array([wma(close_1w[i:i+half_len], half_len) if i+half_len <= n_1w else np.nan 
                            for i in range(n_1w)])
        wma_full = np.array([wma(close_1w[i:i+21], 21) if i+21 <= n_1w else np.nan 
                            for i in range(n_1w)])
        hma_2x_half_minus_full = 2 * wma_half - wma_full
        hma_21 = np.array([wma(hma_2x_half_minus_full[i:i+sqrt_len], sqrt_len) 
                          if i+sqrt_len <= len(hma_2x_half_minus_full) else np.nan 
                          for i in range(len(hma_2x_half_minus_full))])
        
        # Pad to match df_1w length
        hma_21_padded = np.full(n_1w, np.nan)
        start_idx = half_len + sqrt_len - 1
        if start_idx < n_1w:
            end_idx = start_idx + len(hma_21)
            if end_idx <= n_1w:
                hma_21_padded[start_idx:end_idx] = hma_21
        
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_padded)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian(20) channels ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().shift(1).values
    
    # === 1d Indicators: ATR(14) for stoploss ===
    atr_window = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_window, min_periods=atr_window, adjust=False).mean().values
    
    # === 1d Indicators: Volume average (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(donchian_window, 21, atr_window, 20)  # Donchian, HMA, ATR, volume MA warmup
    
    for i in range(warmup, n):
        # --- Skip if any required data is NaN ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # --- Update existing position: check stoploss ---
        if in_position:
            # Stoploss: 2.5 * ATR against position
            if position_side > 0:  # Long position
                if price < entry_price - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if price > entry_price + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i]
        breakout_down = price < donchian_low[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = vol > 1.5 * vol_ma[i]
        
        # HTF trend filter: price relative to 1w HMA(21)
        hma_trend_up = price > hma_21_aligned[i]
        hma_trend_down = price < hma_21_aligned[i]
        
        # Entry conditions: Donchian breakout + volume confirmation + HTF trend alignment
        if breakout_up and volume_confirm and hma_trend_up:
            in_position = True
            position_side = 1
            entry_price = price
            signals[i] = SIZE
        elif breakout_down and volume_confirm and hma_trend_down:
            in_position = True
            position_side = -1
            entry_price = price
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals