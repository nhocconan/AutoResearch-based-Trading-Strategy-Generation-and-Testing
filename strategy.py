#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and low volatility regime.
# Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 AND price > 1d EMA50 AND ATR(14) < 0.5 * ATR(50) (low vol)
# Short when Bear Power > 0 AND price < 1d EMA50 AND ATR(14) < 0.5 * ATR(50) (low vol)
# Uses discrete position size 0.25. Works in both bull and bear markets by using
# 1d EMA50 for trend alignment and low volatility regime to avoid whipsaws.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 6h Indicators: Elder Ray Components ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # ATR for volatility regime
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr_6h).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    
    # === 1d Indicators: EMA50 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for bar-close exit
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(atr_14[i]) or
            np.isnan(atr_50[i]) or np.isnan(ema50_1d_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        bp = bull_power[i]
        br = bear_power[i]
        atr14 = atr_14[i]
        atr50 = atr_50[i]
        ema50 = ema50_1d_aligned[i]
        
        # === EXIT LOGIC (bar-close based) ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power turns negative OR price crosses below 1d EMA50
            if bp <= 0 or price < ema50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power turns negative OR price crosses above 1d EMA50
            if br <= 0 or price > ema50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Low volatility regime: ATR14 < 0.5 * ATR50
            low_vol = atr14 < (0.5 * atr50)
            
            # LONG: Bull Power > 0 AND price > 1d EMA50 AND low volatility
            if bp > 0 and price > ema50 and low_vol:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bear Power > 0 AND price < 1d EMA50 AND low volatility
            elif br > 0 and price < ema50 and low_vol:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_LowVolRegime_V1"
timeframe = "6h"
leverage = 1.0