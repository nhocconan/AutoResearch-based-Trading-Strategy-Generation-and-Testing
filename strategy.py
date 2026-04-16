#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and low volatility regime.
# Long when Bull Power > 0 AND price > 1d EMA50 AND ATR(14) < ATR(50) (low vol regime).
# Short when Bear Power < 0 AND price < 1d EMA50 AND ATR(14) < ATR(50).
# Exit on opposite Elder Ray signal or ATR-based stop (2*ATR from entry).
# Uses discrete position size 0.25. Works in bull/bear by combining momentum, trend, and volatility filters.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 6h Indicators: Elder Ray Index (13-period EMA) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # === 6h ATR for stoploss and volatility regime ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr_6h).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    low_vol_regime = atr_14 < atr_50  # Low volatility when short-term ATR < long-term ATR
    
    # === 1d Indicators: EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(atr_14[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(low_vol_regime[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        ema50 = ema50_1d_aligned[i]
        low_vol = low_vol_regime[i]
        atr_val = atr_14[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bear Power becomes negative (momentum reversal)
            if bear_power[i] >= 0:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bull Power becomes positive (momentum reversal)
            if bull_power[i] <= 0:
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
            # LONG: Bull Power > 0 AND price > 1d EMA50 AND low vol regime
            if bull_power[i] > 0 and price > ema50 and low_vol:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bear Power < 0 AND price < 1d EMA50 AND low vol regime
            elif bear_power[i] < 0 and price < ema50 and low_vol:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_LowVolRegime_V1"
timeframe = "6h"
leverage = 1.0