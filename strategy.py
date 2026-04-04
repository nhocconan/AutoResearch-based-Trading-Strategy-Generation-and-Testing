#!/usr/bin/env python3
"""
Experiment #3399: 6h Elder Ray + 12h ADX Regime Filter
HYPOTHESIS: Elder Ray (Bull/Bear Power) captures institutional buying/selling pressure,
filtered by 12h ADX regime (ADX>25 = trending, ADX<20 = ranging). In trending regimes,
we follow Elder Ray direction (long Bull Power>0, short Bear Power<0). In ranging regimes,
we fade extremes (short when Bull Power>0 & overbought, long when Bear Power<0 & oversold).
Uses 6h timeframe for optimal trade frequency (target: 75-150 trades over 4 years = 19-38/year).
Position size 0.25 with ATR(14) trailing stop (2.0x) to manage risk. Works in both bull
and bear markets by adapting to regime: trend following in strong trends, mean reversion
in choppy/range markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3399_6h_elder_ray_12h_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX regime filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h data
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        # Smoothed values
        tr_period = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        dm_plus_period = pd.Series(dm_plus).rolling(window=period, min_periods=period).mean().values
        dm_minus_period = pd.Series(dm_minus).rolling(window=period, min_periods=period).mean().values
        # Directional Indicators
        di_plus = 100 * dm_plus_period / tr_period
        di_minus = 100 * dm_minus_period / tr_period
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Elder Ray (Bull Power/Bear Power) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === 6h Indicators: RSI(14) for overbought/oversold in ranging markets ===
    def calculate_rsi(close, period=14):
        if len(close) < period + 1:
            return np.full_like(close, np.nan)
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate([[np.nan], rsi])
    
    rsi = calculate_rsi(close, 14)
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, 13, 14, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(rsi[i]) or np.isnan(adx_12h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        adx = adx_12h_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        rsi_val = rsi[i]
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        if adx > 25:
            # TRENDING REGIME: Follow Elder Ray direction
            if bull > 0:  # Strong buying pressure
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif bear < 0:  # Strong selling pressure
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        elif adx < 20:
            # RANGING REGIME: Fade extremes with RSI confirmation
            if bull > 0 and rsi_val > 70:  # Overbought with buying pressure (likely false breakout)
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            elif bear < 0 and rsi_val < 30:  # Oversold with selling pressure (likely false breakdown)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        else:
            # Transition regime (20 <= ADX <= 25): no trade
            signals[i] = 0.0
    
    return signals