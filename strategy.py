#!/usr/bin/env python3
"""
Experiment #023: 12h ADX + RSI + BB Regime Strategy

HYPOTHESIS: 12h timeframe with tight ADX/RSI/BB filters should generate 50-150 trades:
- ADX > 28 = trending regime (stronger than CHOP for filtering)
- RSI(14) < 35 = oversold long entry, > 65 = overbought short entry (not extreme)
- BB(20,2.0) = price structure reference (upper/lower band touch)
- 12h timeframe naturally limits trade count (~365 days/year * 2 = 730 potential bars)
- With ADX>28 filter, expect only 15-20% of bars to qualify = 50-150 trades

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: ADX trending up + RSI oversold pullback + BB lower touch = strong bounce
- Bear: ADX trending down + RSI overbought rally + BB upper touch = short reversal
- ADX works symmetrically for trending conditions in either direction

TARGET: 75-150 total trades over 4 years (~18-37/year) on 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_adx_rsi_bb_trend_v1"
timeframe = "12h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * np.sum(plus_dm[i-period+1:i+1]) / atr[i] / period
            minus_di[i] = 100 * np.sum(minus_dm[i-period+1:i+1]) / atr[i] / period
    
    adx = np.full(n, np.nan)
    dx = np.zeros(n)
    for i in range(period, n):
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx_series = pd.Series(dx)
    adx = adx_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(close, prepend=close[0])
    deltas[0] = 0
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, np.where(avg_loss == 0, 1e-10, avg_loss))
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands - returns upper, middle, lower"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    return upper, middle, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ema(close, period):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for multi-timeframe confirmation
    ema_50_1d = calculate_ema(df_1d['close'].values, period=50)
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Local 12h indicators ===
    adx_14 = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA for confirmation
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
    
    warmup = 150  # 20 for BB + 14 for ADX/RSI + some buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME: ADX > 28 = trending ===
        is_trending = adx_14[i] > 28
        
        # === RSI ZONES ===
        rsi_val = rsi_14[i]
        is_oversold = rsi_val < 35
        is_overbought = rsi_val > 65
        
        # === BB TOUCH ===
        bb_up_touch = close[i] >= bb_upper[i]
        bb_lo_touch = close[i] <= bb_lower[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === HTF TREND ===
        htf_bull = close[i] > ema_aligned[i]
        htf_bear = close[i] < ema_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Trending + oversold + BB lower touch + volume + HTF bull ===
            if is_trending and is_oversold and bb_lo_touch and vol_spike and htf_bull:
                desired_signal = SIZE
            
            # === SHORT: Trending + overbought + BB upper touch + volume + HTF bear ===
            if is_trending and is_overbought and bb_up_touch and vol_spike and htf_bear:
                desired_signal = -SIZE
        
        # === STOPLOSS (3x ATR) ===
        if in_position:
            if position_side > 0:
                stop_price = entry_price - 3.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if ADX drops (trend weakening)
                if adx_14[i] < 22:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                stop_price = entry_price + 3.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if ADX drops
                if adx_14[i] < 22:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals