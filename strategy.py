#!/usr/bin/env python3
"""
Experiment #3415: 6h Elder Ray + Regime Filter (ADX)
HYPOTHESIS: Elder Ray (Bull/Bear Power) captures institutional buying/selling pressure, 
while ADX regime filter (ADX>25) ensures we only trade in trending markets. 
This combination works in bull markets (buy on Bull Power > 0) and bear markets 
(sell on Bear Power < 0) by fading extended moves in ranging conditions (ADX<20).
Primary timeframe 6h balances trade frequency and signal quality.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3415_6h_elderray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for EMA200 trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(200) on 1w close
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === 6h Indicators: EMA13 for Elder Ray (standard setting) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray Components ===
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13  
    bear_power = low - ema13
    
    # === 6h Indicators: ADX(14) for regime detection ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed TR, +DM, -DM
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(atr != 0, atr, np.nan)
    minus_di = 100 * minus_dm_smooth / np.where(atr != 0, atr, np.nan)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.nan)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    
    warmup = max(200, 13, 14)  # sufficient for EMA200w, EMA13, ADX14
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions: regime change or power reversal
            if position_side > 0:  # Long position
                # Exit if Bear Power becomes positive (selling pressure) OR regime turns ranging
                if bear_power[i] > 0 or adx[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit if Bull Power becomes negative (buying pressure) OR regime turns ranging
                if bull_power[i] < 0 or adx[i] < 20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Regime filter: only trade when ADX > 25 (trending market)
        if adx[i] > 25:
            # 1w EMA200 trend filter: align with weekly trend
            weekly_bias = price > ema_200_1w_aligned[i]
            
            # Long entry: Bull Power > 0 (buying pressure) + weekly uptrend bias
            if bull_power[i] > 0 and weekly_bias:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            # Short entry: Bear Power < 0 (selling pressure) + weekly downtrend bias
            elif bear_power[i] < 0 and not weekly_bias:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # In ranging markets (ADX < 25), stay flat to avoid whipsaw
            signals[i] = 0.0
    
    return signals