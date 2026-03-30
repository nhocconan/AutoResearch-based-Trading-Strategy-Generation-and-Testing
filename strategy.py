#!/usr/bin/env python3
"""
Experiment #004: 1h Donchian Pullback + 1w Trend + RSI Filter

HYPOTHESIS: Use 1h candles for more signal opportunities while filtering with 1w/1d structure:
- 1w SMA(50) = macro trend (bull if price> SMA, bear if price< SMA)
- 1d Donchian(20) = intermediate structure (entries on pullback TO channel)
- 1h RSI(14) < 35 for longs, > 65 for shorts = mean-reversion entries
- 1h volume > 1.5x 20-MA = confirmation
- 1d ATR stop = risk management

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Pullbacks to 1d Donchian support + RSI oversold = accumulation
- Bear: Rallies to 1d Donchian resistance + RSI overbought = distribution
- 1w trend filter prevents fighting major direction
- Mean-reversion entries on 1h TF catch shorter-term swings within trends

PRIMARY: 1h | HTF: 1w and 1d
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_pullback_rsi_1w_trend_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """RSI indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_sma(values, period):
    """Simple Moving Average"""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA(50) for macro trend
    sma_50_1w = calculate_sma(df_1w['close'].values, 50)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # 1d Donchian(20) for structure
    donchian_up_1d, donchian_lo_1d = calculate_donchian(df_1d['high'].values, df_1d['low'].values, period=20)
    donchian_up_aligned = align_htf_to_ltf(prices, df_1d, donchian_up_1d)
    donchian_lo_aligned = align_htf_to_ltf(prices, df_1d, donchian_lo_1d)
    
    # === Local 1h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_up_1h, donchian_lo_1h = calculate_donchian(high, low, period=20)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 400  # Extra warmup for 1w SMA(50) alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_lo_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === 1w TREND FILTER ===
        price_vs_1w_sma = close[i] > sma_50_1w_aligned[i]  # True = bull, False = bear
        bull_market = price_vs_1w_sma
        bear_market = not price_vs_1w_sma
        
        # === 1h RSI (mean reversion) ===
        rsi_value = rsi_14[i]
        rsi_oversold = rsi_value < 35  # Long entry
        rsi_overbought = rsi_value > 65  # Short entry
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === 1h DONCHIAN STRUCTURE ===
        # Long: price near 1h Donchian lower (pullback entry)
        # Short: price near 1h Donchian upper (rally entry)
        near_donchian_lo = low[i] <= donchian_lo_1h[i] * 1.02  # Within 2% of lower band
        near_donchian_up = high[i] >= donchian_up_1h[i] * 0.98  # Within 2% of upper band
        
        # === PULLBACK ENTRY (price pulled back to structure) ===
        # In bull: price pulled back from 1h high to near lower band with RSI oversold
        # In bear: price rallied back to upper band with RSI overbought
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Pullback in bull market + RSI oversold + near lower band + volume
            if bull_market and rsi_oversold and near_donchian_lo and vol_spike:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Rally in bear market + RSI overbought + near upper band + volume
            if bear_market and rsi_overbought and near_donchian_up and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS AND EXIT ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # ATR-based trailing stop: 2.5 ATR from recent high
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if trend flips (price crosses below 1w SMA)
                if close[i] < sma_50_1w_aligned[i]:
                    desired_signal = 0.0
                
                # Exit if RSI becomes overbought (momentum exhausted)
                if rsi_value > 70:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # ATR-based trailing stop: 2.5 ATR from recent low
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if trend flips (price crosses above 1w SMA)
                if close[i] > sma_50_1w_aligned[i]:
                    desired_signal = 0.0
                
                # Exit if RSI becomes oversold (momentum exhausted)
                if rsi_value < 30:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 8 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 8:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals