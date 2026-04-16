#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d EMA50 trend filter and low volatility regime.
# Long when Bull Power > 0 AND price > 1d EMA50 AND ATR ratio < 1.2 (low vol).
# Short when Bear Power < 0 AND price < 1d EMA50 AND ATR ratio < 1.2.
# Uses discrete position size 0.25. Works in both bull and bear markets by
# combining trend following (EMA50) with momentum (Elder Ray) and volatility filter.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 6h Indicators: Elder Ray Index (13-period EMA) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h ATR for volatility regime ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_6h).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_6h / atr_ma_50  # Current ATR vs 50-period average
    
    # === 1d Indicators: EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(atr_ratio[i]) or
            np.isnan(ema50_1d_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        ema50 = ema50_1d_aligned[i]
        low_vol = atr_ratio[i] < 1.2  # Low volatility regime
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Elder Ray turns bearish OR price below EMA50 OR high volatility
            if bull_power[i] <= 0 or price < ema50 or not low_vol:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Elder Ray turns bullish OR price above EMA50 OR high volatility
            if bear_power[i] >= 0 or price > ema50 or not low_vol:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power positive AND price above EMA50 AND low volatility
            if bull_power[i] > 0 and price > ema50 and low_vol:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power negative AND price below EMA50 AND low volatility
            elif bear_power[i] < 0 and price < ema50 and low_vol:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_LowVolRegime_V1"
timeframe = "6h"
leverage = 1.0