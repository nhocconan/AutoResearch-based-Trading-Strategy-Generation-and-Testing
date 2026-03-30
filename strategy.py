#!/usr/bin/env python3
"""
Experiment #023: 1d KAMA + RSI(14) Extreme + 1w SMA200 Trend

HYPOTHESIS: KAMA adapts to volatility, catching trend shifts earlier than EMA.
RSI(14) < 30 or > 70 marks true extremes. Weekly SMA200 trend confirmation avoids
counter-trend trades in bear rallies. This is the proven SOL pattern (Sharpe 1.31)
that should generalize to BTC/ETH.

WHY 1d: Lower frequency = fewer trades = less fee drag.
WHY KAMA: Adaptive moving average - faster response than EMA.
WHY RSI EXTREMES: Mean-reversion entries at market extremes = high win rate.
WHY 1w SMA200: Weekly trend filter - avoids fighting major trends.

TARGET: 50-100 total trades over 4 years (12-25/year).
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_sma200_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=14, fast_ema=2, slow_ema=30):
    """Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < slow_ema + 1:
        return np.full(n, np.nan)
    
    # Calculate EMA efficiency ratio
    abs_diff = np.abs(close - np.roll(close, 1))
    abs_diff[0] = 0
    
    # Volatility (sum of price changes over period)
    volatility = pd.Series(abs_diff).rolling(window=period, min_periods=period).sum().values
    
    # Efficiency ratio (0 to 1)
    er = np.zeros(n)
    for i in range(period, n):
        if volatility[i] > 1e-10:
            er[i] = abs_diff[i] / volatility[i]
    
    # Smoothing constants
    fast_const = (2 / (fast_ema + 1)) ** 2
    slow_const = (2 / (slow_ema + 1)) ** 2
    sc = (er * (fast_const - slow_const) + slow_const) ** 2
    
    kama = np.zeros(n)
    kama[period] = close[period]  # Initialize
    
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === Load 1w HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly SMA200 for trend (using close of 1w bars)
    sma200_1w = pd.Series(df_1w['close'].values).rolling(window=200, min_periods=200).mean().values
    sma200_1w_aligned = align_htf_to_ltf(prices, df_1w, sma200_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # KAMA(14) - adaptive trend
    kama_14 = calculate_kama(close, period=14, fast_ema=2, slow_ema=30)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0).ewm(span=14, min_periods=14, adjust=False).mean().values
    loss = (-delta.clip(upper=0)).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = gain / np.where(loss > 1e-10, loss, 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    profit_taken = False
    bars_since_entry = 0
    
    warmup = 250  # Need 200 for weekly SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # Skip if indicators not aligned
        if np.isnan(kama_14[i]) or np.isnan(sma200_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend direction
        price_above_1w_sma = close[i] > sma200_1w_aligned[i]
        
        # KAMA trend direction
        kama_trend_up = kama_14[i] > kama_14[i - 1] if i > 0 else False
        
        # RSI extremes
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        rsi_extreme = rsi_14[i] < 25 or rsi_14[i] > 75  # Stricter for entries
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Weekly uptrend + KAMA rising + RSI extreme oversold
            if price_above_1w_sma and kama_trend_up and rsi_extreme:
                desired_signal = SIZE
            
            # SHORT: Weekly downtrend + KAMA falling + RSI extreme overbought
            if not price_above_1w_sma and not kama_trend_up and rsi_extreme:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position and position_side > 0:
            if close[i] < entry_price - 2.5 * entry_atr:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if close[i] > entry_price + 2.5 * entry_atr:
                desired_signal = 0.0
        
        # === TAKE PROFIT at 2R + half position ===
        bars_since_entry = i - entry_bar
        if in_position and not profit_taken and bars_since_entry >= 2:
            if position_side > 0:
                profit_2r = entry_price + 2.0 * entry_atr
                if high[i] >= profit_2r:
                    desired_signal = SIZE * 0.5  # Take half profit
                    profit_taken = True
            elif position_side < 0:
                profit_2r = entry_price - 2.0 * entry_atr
                if low[i] <= profit_2r:
                    desired_signal = -SIZE * 0.5
                    profit_taken = True
        
        # === HOLD MINIMUM 2 BARS to avoid fee churn ===
        if in_position and bars_since_entry < 2:
            desired_signal = SIZE if position_side > 0 else -SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction change
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                profit_taken = False
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals