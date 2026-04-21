#!/usr/bin/env python3
"""
6h_1d_LiquidityVoid_BullBearBalance
Hypothesis: Combines liquidity voids (fair value gaps) with bull/bear power to identify high-probability mean reversion and continuation setups. Voids act as magnets for price, while bull/bear power confirms institutional participation. Designed for low trade frequency (target: 12-37/year) with strong risk control. Works in bull/bear regimes by adapting to market structure via imbalance detection.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for bull/bear power and liquidity voids
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Calculate daily EMA13 for bull/bear power
    close_series = pd.Series(close_daily)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_daily - ema13
    bear_power = ema13 - low_daily
    
    # Align daily indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_daily, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_daily, bear_power)
    
    # Identify liquidity voids (Fair Value Gaps) on daily timeframe
    # Bullish FVG: gap between low[i-1] and high[i+1] where low[i-1] > high[i+1]
    # Bearish FVG: gap between high[i-1] and low[i+1> where high[i-1] < low[i+1]
    fvg_bull = np.zeros(len(high_daily), dtype=bool)
    fvg_bear = np.zeros(len(high_daily), dtype=bool)
    
    for i in range(1, len(high_daily) - 1):
        # Bullish FVG: previous low > next high
        if low_daily[i-1] > high_daily[i+1]:
            fvg_bull[i] = True
        # Bearish FVG: previous high < next low
        if high_daily[i-1] < low_daily[i+1]:
            fvg_bear[i] = True
    
    # Align FVG signals to 6h timeframe
    fvg_bull_aligned = align_htf_to_ltf(prices, df_daily, fvg_bull.astype(float))
    fvg_bear_aligned = align_htf_to_ltf(prices, df_daily, fvg_bear.astype(float))
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period average (less strict to avoid over-filtering)
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(fvg_bull_aligned[i]) or np.isnan(fvg_bear_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        bp = bull_power_aligned[i]
        be = bear_power_aligned[i]
        fvg_bull_signal = fvg_bull_aligned[i] > 0.5
        fvg_bear_signal = fvg_bear_aligned[i] > 0.5
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long setup: bullish FVG with bullish power and volume
            # Price should be in or above the bullish FVG zone
            if fvg_bull_signal and bp > 0 and vol_ok:
                # Additional confirmation: bullish candle (close > open)
                if close[i] > prices['open'].values[i]:
                    signals[i] = 0.25
                    position = 1
            # Short setup: bearish FVG with bearish power and volume
            elif fvg_bear_signal and be > 0 and vol_ok:
                # Additional confirmation: bearish candle (close < open)
                if close[i] < prices['open'].values[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: bearish FVG appears or bullish power fails
            if fvg_bear_signal or bp <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish FVG appears or bearish power fails
            if fvg_bull_signal or be <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_LiquidityVoid_BullBearBalance"
timeframe = "6h"
leverage = 1.0