#!/usr/bin/env python3
# 12h_hma_volatility_breakout_v1
# Hypothesis: 12h strategy using HMA(21) as dynamic trend filter with volatility-based breakout detection. Enters long when price breaks above HMA(21) + 0.5*ATR(14) with volume confirmation (>1.3x 20-period average) and bullish 1d trend (price > 20-period EMA on 1d). Enters short when price breaks below HMA(21) - 0.5*ATR(14) with volume confirmation and bearish 1d trend. Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 15-35 trades/year) by requiring volatility expansion and HTF alignment, reducing whipsaws in ranging markets while capturing strong trends in both bull and bear regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma1 = pd.Series(series).ewm(span=half_period, adjust=False, min_periods=half_period).mean()
    wma2 = pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean()
    return hma.values

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift(1)).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr.values

name = "12h_hma_volatility_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # HMA(21) on primary timeframe
    hma_21 = calculate_hma(close, 21)
    
    # ATR(14) for volatility-based breakout bands
    atr_14 = calculate_atr(high, low, close, 14)
    
    # 1d HTF trend filter: 20-period EMA on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(hma_21[i]) or
            np.isnan(atr_14[i]) or np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Dynamic breakout levels
        upper_breakout = hma_21[i] + 0.5 * atr_14[i]
        lower_breakout = hma_21[i] - 0.5 * atr_14[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below HMA (trend reversal)
            if close[i] < hma_21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above HMA (trend reversal)
            if close[i] > hma_21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation, volatility breakout, and 1d trend alignment
            if volume_confirmed:
                # Bullish 1d trend: price above 20-period EMA
                bullish_trend = close[i] > ema_20_1d_aligned[i]
                # Bearish 1d trend: price below 20-period EMA
                bearish_trend = close[i] < ema_20_1d_aligned[i]
                
                # Long: price breaks above HMA + 0.5*ATR with volume and bullish 1d trend
                if close[i] > upper_breakout and bullish_trend:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below HMA - 0.5*ATR with volume and bearish 1d trend
                elif close[i] < lower_breakout and bearish_trend:
                    position = -1
                    signals[i] = -0.25
    
    return signals