#!/usr/bin/env python3
"""
Experiment #028: 1d TRIX Momentum + Weekly SMA200 + ADX Regime Filter

HYPOTHESIS: TRIX crossover is proven on 4h (ETHUSDT 1.32 Sharpe). Applying to 1d
reduces trades by ~6x, and adding 1w SMA200 gives structural trend direction.
ADX>25 filter avoids 2022 range-bound whipsaws where momentum oscillators fail.

WHY IT WORKS IN BULL AND BEAR:
- Bull: TRIX crosses from negative to positive + price above weekly SMA200
- Bear: TRIX crosses from positive to negative + price below weekly SMA200
- ADX filter keeps positions flat when ADX<20 (choppy, no trend)

TARGET: 30-100 total trades over 4 years = 7-25/year. HARD MAX: 150.
Signal size: 0.25 (conservative).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_trix_adx_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_trix(close, period=9):
    """TRIX - Triple EMA rate of change. Momentum indicator."""
    close_s = pd.Series(close, dtype=np.float64)
    ema1 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    # Rate of change of triple EMA
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    return trix.fillna(0).values

def calculate_adx(high, low, close, period=14):
    """ADX - Average Directional Index. Returns (adx, plus_di, minus_di)."""
    high_s = pd.Series(high, dtype=np.float64)
    low_s = pd.Series(low, dtype=np.float64)
    close_s = pd.Series(close, dtype=np.float64)
    
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di > 0, plus_di + minus_di, 1)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly SMA200 for trend direction
    weekly_sma = pd.Series(df_1w['close'].values).rolling(window=200, min_periods=200).mean().values
    weekly_sma_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma)
    
    # Local indicators
    trix = calculate_trix(close, period=9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    warmup = 300  # Need 200 for weekly SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(adx[i]) or np.isnan(trix[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(weekly_sma_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1w SMA200) ===
        weekly_bull = close[i] > weekly_sma_aligned[i]
        
        # === REGIME (ADX) ===
        # ADX > 25 = trending (trade), ADX < 20 = ranging (flat)
        is_trending = adx[i] > 25.0
        is_ranging = adx[i] < 20.0
        
        # Previous TRIX values for crossover detection
        prev_trix = trix[i - 1] if i > 0 else 0.0
        prev_prev_trix = trix[i - 2] if i > 1 else prev_trix
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position and is_trending:
            # === LONG: TRIX crosses above zero (momentum shift) ===
            # Requires: prev_prev_trix < 0 <= prev_trix (crossing up)
            if prev_prev_trix < 0 and prev_trix >= 0:
                if weekly_bull:
                    desired_signal = SIZE
            
            # === SHORT: TRIX crosses below zero (momentum shift) ===
            if prev_prev_trix > 0 and prev_trix <= 0:
                if not weekly_bull:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT ON MOMENTUM REVERSAL ===
        if in_position and desired_signal == 0.0:
            # TRIX crosses back through zero = momentum reversal
            if position_side > 0 and prev_trix > 0 and trix[i] <= 0:
                desired_signal = 0.0
            if position_side < 0 and prev_trix < 0 and trix[i] >= 0:
                desired_signal = 0.0
        
        # === RANGING MARKET EXIT ===
        # If ADX drops below 20 while in position, exit (trend exhausted)
        if in_position and is_ranging:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # ATR-based stoploss (2.5x ATR, minimum 2%)
                min_stop = entry_price * 0.02
                atr_stop = 2.5 * entry_atr
                if position_side > 0:
                    stop_price = entry_price - max(atr_stop, min_stop)
                else:
                    stop_price = entry_price + max(atr_stop, min_stop)
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals