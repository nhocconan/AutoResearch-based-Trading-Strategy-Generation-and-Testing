#!/usr/bin/env python3
"""
Experiment #1939: 6h Elder Ray + 12h Regime Filter + Volume Spike
HYPOTHESIS: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure. 
Combined with 12h ADX regime filter (trending vs ranging) and volume confirmation to avoid false signals.
In trending markets (ADX>25): follow Elder Ray direction. In ranging markets (ADX<20): fade extreme readings.
Target: 75-150 total trades over 4 years with strict entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1939_6h_elder_ray_12h_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for regime filter (ADX) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX(14) for regime detection
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Elder Ray (Bull/Bear Power) and EMA ===
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 6h volume spike
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA(13), volume MA, and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_13[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions: Elder Ray divergence or adverse move
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if Bear Power turns positive (selling pressure)
                if bear_power[i] > 0:
                    exit_signal = True
                # Exit if price closes below EMA(13)
                elif price < ema_13[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if Bull Power turns negative (buying pressure)
                if bull_power[i] < 0:
                    exit_signal = True
                # Exit if price closes above EMA(13)
                elif price > ema_13[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        adx_val = adx_12h_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            if adx_val > 25:  # Trending market - follow Elder Ray
                # Long: Bull Power > 0 and rising (increasing buying pressure)
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: Bear Power < 0 and falling (increasing selling pressure)
                elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif adx_val < 20:  # Ranging market - fade extreme readings
                # Long: Bear Power extremely negative (oversold) and turning up
                if bear_power[i] < np.percentile(bear_power[max(0, i-100):i+1], 5) and bear_power[i] > bear_power[i-1]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: Bull Power extremely positive (overbought) and turning down
                elif bull_power[i] > np.percentile(bull_power[max(0, i-100):i+1], 95) and bull_power[i] < bull_power[i-1]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:  # Transition regime (ADX 20-25) - no trades
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals