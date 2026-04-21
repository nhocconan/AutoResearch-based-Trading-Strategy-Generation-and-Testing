#!/usr/bin/env python3
"""
6h_WilliamsAlligator_ADX_Regime_v1
Hypothesis: Combines Williams Alligator (trend detection) with ADX regime filter on 6h timeframe.
Uses 1d HTF for trend bias and 1w HTF for major regime filter. Designed for low trade frequency
(~15-30/year) to minimize fee drag while capturing major trends in both bull and bear markets.
Alligator identifies trend direction, ADX filters for strong trends, and HTF context prevents
counter-trend trading during major regime shifts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 60 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d trend bias: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1w regime filter: 50-period EMA (major trend) ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Williams Alligator on 6h (primary timeframe) ===
    # Alligator consists of three smoothed moving averages (SMMA)
    median_price = (prices['high'].values + prices['low'].values) / 2
    
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    def smma(source, period):
        """Smoothed Moving Average"""
        result = np.full_like(source, np.nan, dtype=float)
        if len(source) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + PRICE) / PERIOD
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward
    teeth = smma(median_price, 8)
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    lips = smma(median_price, 5)
    
    # Apply Alligator shifts (forward shift means we use past values)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted values with NaN
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # === ADX (14-period) for trend strength ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions
        # Bullish alignment: Lips > Teeth > Jaw
        bull_alligator = lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i]
        # Bearish alignment: Jaw > Teeth > Lips
        bear_alligator = jaw_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > lips_shifted[i]
        
        # ADX trend strength filter (>25 = strong trend)
        strong_trend = adx[i] > 25
        
        # HTF regime filters
        # 1d trend bias: price above/below EMA34
        price_above_1d_trend = prices['close'].values[i] > ema_34_1d_aligned[i]
        price_below_1d_trend = prices['close'].values[i] < ema_34_1d_aligned[i]
        
        # 1w major regime: price above/below EMA50
        price_above_1w_regime = prices['close'].values[i] > ema_50_1w_aligned[i]
        price_below_1w_regime = prices['close'].values[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Bullish Alligator + strong trend + price above 1d EMA + price above 1w EMA
            if bull_alligator and strong_trend and price_above_1d_trend and price_above_1w_regime:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + strong trend + price below 1d EMA + price below 1w EMA
            elif bear_alligator and strong_trend and price_below_1d_trend and price_below_1w_regime:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Alligator reversal OR ADX weakens OR price crosses 1d EMA opposite
            if position == 1:
                # Exit long: bearish Alligator OR weak trend OR price below 1d EMA
                if (not bull_alligator) or (adx[i] < 20) or (not price_above_1d_trend):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: bullish Alligator OR weak trend OR price above 1d EMA
                if (not bear_alligator) or (adx[i] < 20) or (not price_below_1d_trend):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ADX_Regime_v1"
timeframe = "6h"
leverage = 1.0