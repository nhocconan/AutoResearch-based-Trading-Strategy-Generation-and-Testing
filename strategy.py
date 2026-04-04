#!/usr/bin/env python3
"""
Experiment #3819: 6h Elder Ray + 12h ADX Regime Filter + Volume Spike
HYPOTHESIS: Combines Elder Ray (bull/bear power) with 12h ADX regime filter to trade with momentum in trending markets (ADX>25) and mean-revert in ranging markets (ADX<20). Volume spike (>1.5x 20-bar MA) confirms institutional participation. Designed for 6h timeframe to balance trade frequency and signal quality. Works in both bull (long bias when bull power > 0) and bear (short bias when bear power < 0) markets. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3819_6h_elder_ray_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX regime filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX(14)
    def true_range(high, low, prev_close):
        return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    
    prev_close_12h = np.roll(close_12h, 1)
    prev_close_12h[0] = close_12h[0]
    tr_12h = true_range(high_12h, low_12h, prev_close_12h)
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = np.diff(low_12h, prepend=low_12h[0]) * -1  # positive values
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    atr_smooth = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / np.where(atr_smooth == 0, 1, atr_smooth)
    minus_di = 100 * minus_dm_smooth / np.where(atr_smooth == 0, 1, atr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 12h ADX to 6h timeframe (shifted by 1 for completed 12h bar)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Elder Ray (Bull/Bear Power) ===
    # EMA(13) as proxy for fair value
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13 (typically negative)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    
    warmup = max(20, 14)  # sufficient for volume MA and EMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Fixed stoploss: 2.5 * ATR(14) from entry
            atr_14 = pd.Series(
                np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
            ).rolling(window=14, min_periods=14).mean().values[i]
            
            if position_side > 0:  # Long
                if price < stop_price:  # Stoploss hit
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > stop_price:  # Stoploss hit
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            adx = adx_12h_aligned[i]
            
            if adx > 25:  # Trending regime - trade with momentum
                # Long: Bull Power > 0 (strong buying pressure)
                if bull_power[i] > 0:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    atr_14 = pd.Series(
                        np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
                    ).rolling(window=14, min_periods=14).mean().values[i]
                    stop_price = entry_price - 2.5 * atr_14
                    signals[i] = SIZE
                # Short: Bear Power < 0 (strong selling pressure)
                elif bear_power[i] < 0:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    atr_14 = pd.Series(
                        np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
                    ).rolling(window=14, min_periods=14).mean().values[i]
                    stop_price = entry_price + 2.5 * atr_14
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif adx < 20:  # Ranging regime - mean revert at extremes
                # Long: Bear Power < 0 and price near low (oversold)
                if bear_power[i] < 0 and low[i] < ema_13[i] - 0.5 * np.std(close[max(0, i-50):i+1]):
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    atr_14 = pd.Series(
                        np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
                    ).rolling(window=14, min_periods=14).mean().values[i]
                    stop_price = entry_price - 2.5 * atr_14
                    signals[i] = SIZE
                # Short: Bull Power > 0 and price near high (overbought)
                elif bull_power[i] > 0 and high[i] > ema_13[i] + 0.5 * np.std(close[max(0, i-50):i+1]):
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    atr_14 = pd.Series(
                        np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
                    ).rolling(window=14, min_periods=14).mean().values[i]
                    stop_price = entry_price + 2.5 * atr_14
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:  # Transition regime (20 <= ADX <= 25) - no trade
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals