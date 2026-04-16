#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and EMA200 trend filter.
# Long when Bull Power > 0, ADX > 25 (trending), and close > EMA200 (1d).
# Short when Bear Power < 0, ADX > 25 (trending), and close < EMA200 (1d).
# Uses ATR-based stoploss (2*ATR) and discrete position size 0.25.
# Elder Ray measures bull/bear strength relative to EMA13; ADX filters for trending markets.
# Works in both bull and bear markets by requiring trend alignment (EMA200) and volume/volatility confirmation via ADX.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 6h Indicators: EMA13 for Elder Ray, ATR for stoploss ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # True Range and ATR
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1d Indicators: EMA200, ADX (for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA200 (1d)
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # ADX (1d) - using Welles Wilder's method
    plus_dm = pd.Series(high_1d).diff()
    minus_dm = pd.Series(low_1d).diff().abs()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr_1d = pd.concat([
        pd.Series(high_1d).diff(),
        pd.Series(low_1d).diff().abs(),
        pd.Series(close_1d).shift(1).diff().abs()
    ], axis=1).max(axis=1)
    
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di_1d = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    minus_di_1d = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 200 periods needed for EMA200)
    warmup = 200
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema13[i]) or np.isnan(atr_6h[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        bull_power = high[i] - ema13[i]
        bear_power = low[i] - ema13[i]
        adx_val = adx_1d_aligned[i]
        ema200_val = ema200_1d_aligned[i]
        atr_val = atr_6h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bear Power becomes negative (momentum shift)
            if bear_power >= 0:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bull Power becomes positive (momentum shift)
            if bull_power <= 0:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0, ADX > 25 (trending), close > EMA200 (1d)
            if bull_power > 0 and adx_val > 25.0 and price > ema200_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bear Power < 0, ADX > 25 (trending), close < EMA200 (1d)
            elif bear_power < 0 and adx_val > 25.0 and price < ema200_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dADX_EMA200_TrendFilter_V1"
timeframe = "6h"
leverage = 1.0