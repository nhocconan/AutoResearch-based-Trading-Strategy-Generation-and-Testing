#!/usr/bin/env python3
"""
Experiment #2927: 6h Elder Ray + ADX Regime Filter
HYPOTHESIS: Elder Ray (Bull Power/Bear Power) identifies trend strength, while ADX regime filter (ADX>25 = trending, ADX<20 = ranging) prevents whipsaws. In trending regimes (ADX>25), take Elder Ray signals: long when Bull Power>0 and Bear Power<0, short when Bear Power>0 and Bull Power<0. In ranging regimes (ADX<20), fade extremes: long when Bear Power<-0.5*ATR and price>SMA(50), short when Bull Power>0.5*ATR and price<SMA(50). 6h timeframe balances signal quality and trade frequency. Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2927_6h_elder_ray_adx_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 6h Indicators: EMA(13) for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: ATR(14) for volatility ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Bull Power and Bear Power ===
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h Indicators: ADX(14) for regime detection ===
    # +DM and -DM
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed +DM, -DM, TR
    tr_smooth = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: SMA(50) for mean reversion anchor ===
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 60  # sufficient for all indicators (max 50, 14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(atr[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(adx[i]) or np.isnan(sma50[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based trailing stop ---
        if in_position:
            if position_side > 0:  # Long
                # Exit if price drops 2.5*ATR below entry or reverses strongly
                if price < entry_price - 2.5 * atr[i] or bear_power[i] > 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit if price rises 2.5*ATR above entry or reverses strongly
                if price > entry_price + 2.5 * atr[i] or bull_power[i] < 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Regime detection: ADX > 25 = trending, ADX < 20 = ranging
        if adx[i] > 25.0:  # Trending regime
            # Trend following: Elder Ray signals
            if bull_power[i] > 0 and bear_power[i] < 0:  # Strong bullish
                in_position = True
                position_side = 1
                entry_price = price
                signals[i] = SIZE
            elif bear_power[i] > 0 and bull_power[i] < 0:  # Strong bearish
                in_position = True
                position_side = -1
                entry_price = price
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        elif adx[i] < 20.0:  # Ranging regime
            # Mean reversion: fade extremes near SMA50
            if bear_power[i] < -0.5 * atr[i] and price > sma50[i]:  # Oversold bounce long
                in_position = True
                position_side = 1
                entry_price = price
                signals[i] = SIZE
            elif bull_power[i] > 0.5 * atr[i] and price < sma50[i]:  # Overbought fade short
                in_position = True
                position_side = -1
                entry_price = price
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:  # Transition regime (ADX 20-25) - stay flat
            signals[i] = 0.0
    
    return signals