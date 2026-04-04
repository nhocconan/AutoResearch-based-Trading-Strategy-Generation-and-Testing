#!/usr/bin/env python3
"""
Experiment #4621: 4h Donchian(20) Breakout + HMA Trend + Volume Confirmation
HYPOTHESIS: 4h price breaking Donchian(20) channels with HMA(21) trend alignment and volume confirmation (>1.3x avg) captures strong momentum. Uses 1d/1w HTF for regime filtering (only trade in bull/bear regimes, avoid chop). Discrete sizing (0.30) and ATR stoploss (2.5x) manage risk. Target: 19-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4621_4h_donchian20_hma_vol_1d_1w_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d and 1w data for regime filtering
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: EMA(50) for bull/bear regime ===
    if len(df_1d) >= 1:
        ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_50_1d = np.array([])
    
    # === 1w Indicators: EMA(20) for long-term trend ===
    if len(df_1w) >= 1:
        ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    else:
        ema_20_1w = np.array([])
    
    # Align HTF indicators to 4h timeframe
    if len(ema_50_1d) > 0:
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
        ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
        ema_20_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) channels ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: HMA(21) for trend ===
    def hull_moving_average(arr, period):
        """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        
        if len(wma_half) == 0 or len(wma_full) == 0:
            return np.full_like(arr, np.nan)
        
        # Align arrays: wma_half starts at half_period-1, wma_full at period-1
        # We need same length, so pad wma_full at beginning
        diff = len(wma_half) - len(wma_full)
        if diff > 0:
            wma_full_padded = np.concatenate([np.full(diff, np.nan), wma_full])
        else:
            wma_full_padded = wma_full[-len(wma_half):] if len(wma_half) <= len(wma_full) else wma_full
            wma_half = wma_half[-len(wma_full_padded):] if len(wma_half) > len(wma_full_padded) else wma_half
            if len(wma_half) < len(wma_full_padded):
                wma_half = np.concatenate([np.full(len(wma_full_padded) - len(wma_half), np.nan), wma_half])
        
        raw_hma = 2 * wma_half - wma_full_padded
        return wma(raw_hma, sqrt_period)
    
    hma_21 = hull_moving_average(close, 21)
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 21, 14)  # Donchian, HMA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(hma_21[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- Regime Filter: Only trade in clear bull/bear markets ---
        # Bull: price > 1d EMA50 AND price > 1w EMA20
        # Bear: price < 1d EMA50 AND price < 1w EMA20
        # Avoid chop: when regime is mixed
        bull_regime = price > ema_50_1d_aligned[i] and price > ema_20_1w_aligned[i]
        bear_regime = price < ema_50_1d_aligned[i] and price < ema_20_1w_aligned[i]
        in_clear_regime = bull_regime or bear_regime
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.3x avg volume)
        vol_confirm = vol_ratio[i] > 1.3
        
        # Donchian breakout conditions
        breakout_long = price > highest_20[i] and hma_21[i] > hma_21[i-1]  # HMA rising
        breakout_short = price < lowest_20[i] and hma_21[i] < hma_21[i-1]  # HMA falling
        
        # Enter long in bull regime on upward breakout with volume
        if bull_regime and breakout_long and vol_confirm:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        # Enter short in bear regime on downward breakout with volume
        elif bear_regime and breakout_short and vol_confirm:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals