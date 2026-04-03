#!/usr/bin/env python3
"""
Experiment #213: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, filtered by 12h Hull Moving Average trend 
(price > HMA = bullish bias, price < HMA = bearish bias), volume spikes (>2.0x average), and 
ATR-based stoploss capture strong momentum moves with reduced false breakouts. The 12h HMA 
provides higher-timeframe trend alignment to avoid counter-trend trades. 4h timeframe targets 
19-50 trades/year (75-200 total over 4 years) to minimize fee drag while capturing significant 
moves. Works in both bull (breakouts with volume) and bear (failed breaks reverse sharply).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_213_4h_donchian_12h_hma_volume_v1"
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
    
    # Calculate Hull Moving Average (HMA) on 12h close
    # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    hma_12h = np.full(len(df_12h), np.nan)
    close_12h = df_12h['close'].values
    n_12h = len(close_12h)
    
    if n_12h >= 16:  # Need enough data for HMA(16)
        half_n = 8
        sqrt_n = int(np.sqrt(16))  # 4
        
        # WMA(close, 8)
        wma_half = np.full(n_12h, np.nan)
        for i in range(half_n, n_12h):
            wma_half[i] = np.mean(close_12h[i-half_n+1:i+1] * np.arange(1, half_n+1)) * 2
        
        # WMA(close, 16)
        wma_full = np.full(n_12h, np.nan)
        for i in range(16, n_12h):
            wma_full[i] = np.mean(close_12h[i-16+1:i+1] * np.arange(1, 17))
        
        # HMA = WMA(2*WMA(8) - WMA(16), 4)
        diff = 2 * wma_half - wma_full
        for i in range(sqrt_n, n_12h):
            if not np.isnan(diff[i]):
                hma_12h[i] = np.mean(diff[i-sqrt_n+1:i+1] * np.arange(1, sqrt_n+1))
    
    # Align HMA to LTF (4h) timeframe with shift(1) for completed bars only
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
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
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF HMA, ATR, and Donchian
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- HMA Trend Filter: Price > HMA = bullish bias, Price < HMA = bearish bias ---
        price_above_hma = close[i] > hma_12h_aligned[i]
        price_below_hma = close[i] < hma_12h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] < donchian_m[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] > donchian_m[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike + price above 12h HMA
        long_condition = breakout_up and volume_spike and price_above_hma
        
        # Short: Donchian breakout down + volume spike + price below 12h HMA
        short_condition = breakout_down and volume_spike and price_below_hma
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
}