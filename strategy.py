#!/usr/bin/env python3
"""
6h_IBS_Regime_Filter_ATR_Volume
Hypothesis: Combine Intraday Bar Strength (IBS) with volatility regime (ATR ratio) and volume confirmation on 6h timeframe.
Long when IBS < 0.3 (oversold) in low volatility regime (ATR(7)/ATR(30) < 0.8) with volume > 1.5x MA.
Short when IBS > 0.7 (overbought) in low volatility regime with volume > 1.5x MA.
Uses 1d HTF for trend filter: only long when price > 1d EMA50, short when price < 1d EMA50.
ATR-based stoploss (2.5x) and discrete sizing (0.25). Targets 80-160 total trades over 4 years (20-40/year).
Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets by combining mean reversion with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Intraday Bar Strength (IBS) = (close - low) / (high - low)
    ibs = (close - low) / (high - low)
    ibs = np.where((high - low) == 0, 0.5, ibs)  # avoid division by zero
    
    # ATR ratio for volatility regime: ATR(7) / ATR(30)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr7 = tr.rolling(window=7, min_periods=7).mean().values
    atr30 = tr.rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr7 / atr30
    
    # Volume filter: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    atr14 = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ibs[i]) or np.isnan(atr_ratio[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ibs_val = ibs[i]
        vol_ratio = atr_ratio[i]
        vol_avg = vol_ma[i]
        ema_50 = ema_50_1d_aligned[i]
        atr = atr14[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume_now > 1.5 * vol_avg
        
        # Low volatility regime: ATR ratio < 0.8 (volatile markets have ratio > 1.2)
        low_vol_regime = vol_ratio < 0.8
        
        if position == 0:
            # Enter only in low volatility with volume confirmation and trend alignment
            long_condition = (ibs_val < 0.3) and low_vol_regime and volume_confirm and (price > ema_50)
            short_condition = (ibs_val > 0.7) and low_vol_regime and volume_confirm and (price < ema_50)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.5x ATR) or trend reversal
            if price < entry_price - 2.5 * atr:
                signals[i] = 0.0
                position = 0
            # Exit long if price falls below EMA50 (trend reversal)
            elif price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.5x ATR) or trend reversal
            if price > entry_price + 2.5 * atr:
                signals[i] = 0.0
                position = 0
            # Exit short if price rises above EMA50 (trend reversal)
            elif price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IBS_Regime_Filter_ATR_Volume"
timeframe = "6h"
leverage = 1.0