#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and ATR-based volatility regime.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) AND price > 1d EMA50 (uptrend) AND ATR(14) < ATR(50) (low volatility regime).
# Short when Bear Power < 0 AND Bull Power > 0 AND price < 1d EMA50 (downtrend) AND ATR(14) < ATR(50).
# Uses discrete position size 0.25. Elder Ray measures bull/bear power relative to EMA13, 1d EMA50 ensures higher timeframe alignment,
# low volatility regime (ATR contraction) filters for breakout-prone conditions. Designed to capture explosive moves in both bull and bear markets.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag while avoiding overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: EMA13 for Elder Ray calculation ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === 6h Indicators: Bull Power and Bear Power (Elder Ray) ===
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # === 6h Indicators: ATR for volatility regime filter ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    low_volatility = atr14 < atr50  # ATR contraction regime
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr14[i]) or np.isnan(atr50[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        bp = bull_power[i]
        br = bear_power[i]
        price = close[i]
        ema_1d = ema_50_1d_aligned[i]
        vol_regime = low_volatility[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power turns negative (loss of bullish momentum) or volatility expands
            if bp <= 0 or not vol_regime:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power turns positive (loss of bearish momentum) or volatility expands
            if br >= 0 or not vol_regime:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 (bullish bias) AND price > 1d EMA50 (uptrend) AND low volatility regime
            if bp > 0 and br < 0 and price > ema_1d and vol_regime:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power < 0 AND Bull Power > 0 (bearish bias) AND price < 1d EMA50 (downtrend) AND low volatility regime
            elif br < 0 and bp > 0 and price < ema_1d and vol_regime:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_LowVolRegime_V1"
timeframe = "6h"
leverage = 1.0