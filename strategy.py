#!/usr/bin/env python3
"""
4h_PriceAction_Reversal_Pattern
Hypothesis: Price action reversal patterns (pin bars, inside bars) at key support/resistance levels 
combined with volume confirmation work in both bull and bear markets by capturing reversals 
at exhaustion points. Uses 1d trend filter for higher timeframe bias and avoids chop via 
volatility regime filter. Target: 20-40 trades/year per symbol.
"""

name = "4h_PriceAction_Reversal_Pattern"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volatility regime filter: ATR ratio (short/long)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr_short = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_long = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_short / np.where(atr_long == 0, np.nan, atr_long)
    # Low volatility (range market) = mean reversion, High volatility = trend
    # We want low volatility for mean reversion reversals
    low_vol = atr_ratio < 0.8  # regime filter
    
    # Volume confirmation: volume > average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma
    
    # Price action patterns
    # Pin bar: long wick, small body
    body_size = np.abs(close - open_price)
    upper_wick = high - np.maximum(open_price, close)
    lower_wick = np.minimum(open_price, close) - low
    is_pin_bar = ((upper_wick > 2 * body_size) | (lower_wick > 2 * body_size)) & (body_size > 0)
    
    # Inside bar: current high < previous high and current low > previous low
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    is_inside_bar = (high < prev_high) & (low > prev_low)
    
    # Support/resistance levels from 1d pivots (simplified: recent swing high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d trend filter: price vs 20-period EMA
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_1d = df_1d['close'].values > ema_20_1d
    downtrend_1d = df_1d['close'].values < ema_20_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Dynamic support/resistance: recent 10-period high/low from 1d data
    # We'll use the 1d high/low as reference levels, aligned to 4h
    resistance_1d = df_1d['high'].values
    support_1d = df_1d['low'].values
    resistance_aligned = align_htf_to_ltf(prices, df_1d, resistance_1d)
    support_aligned = align_htf_to_ltf(prices, df_1d, support_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip first roll values
        if np.isnan(atr_ratio[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
            
        # Only trade in low volatility regime for mean reversion
        if not low_vol[i]:
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Volume confirmation required
        if not vol_confirm[i]:
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Check for price action patterns at key levels
        pin_bar = is_pin_bar[i]
        inside_bar = is_inside_bar[i]
        
        # Distance to support/resistance (normalized by ATR)
        atr_val = atr_short[i] if not np.isnan(atr_short[i]) else 1.0
        dist_to_resistance = (resistance_aligned[i] - high[i]) / atr_val if not np.isnan(resistance_aligned[i]) else 999
        dist_to_support = (low[i] - support_aligned[i]) / atr_val if not np.isnan(support_aligned[i]) else 999
        
        # Consider level touched if within 0.5 ATR
        near_resistance = dist_to_resistance < 0.5
        near_support = dist_to_support < 0.5
        
        if position == 0:
            # LONG: bullish pin bar or inside bar breakout near support in 1d uptrend
            if (pin_bar or inside_bar) and near_support and uptrend_1d_aligned[i]:
                # Additional confirmation: bullish pin bar (long lower wick) or breakout above inside bar high
                if pin_bar and lower_wick[i] > upper_wick[i]:
                    signals[i] = 0.25
                    position = 1
                elif inside_bar and close[i] > prev_high[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: bearish pin bar or inside bar breakdown near resistance in 1d downtrend
            elif (pin_bar or inside_bar) and near_resistance and downtrend_1d_aligned[i]:
                # Additional confirmation: bearish pin bar (long upper wick) or breakdown below inside bar low
                if pin_bar and upper_wick[i] > lower_wick[i]:
                    signals[i] = -0.25
                    position = -1
                elif inside_bar and close[i] < prev_low[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # EXIT LONG: price breaks below support or pattern fails
            if close[i] < support_aligned[i] or (pin_bar and upper_wick[i] > lower_wick[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above resistance or pattern fails
            if close[i] > resistance_aligned[i] or (pin_bar and lower_wick[i] > upper_wick[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals