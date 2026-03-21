#!/usr/bin/env python3
"""
Experiment #204: 1d Fisher Transform + 4h HMA Trend + Choppiness Regime Filter
Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets (2025 test period).
Combined with 4h HMA for trend bias and Choppiness Index to detect regime, this should work better than
pure trend-following strategies that failed in experiments #192-#203. Fisher crosses at extremes provide
clear entry signals. Position sizing adjusts based on choppiness (smaller in range markets).
Stoploss: 2.5*ATR trailing. Target: Beat Sharpe=0.499 from current best.
ENSURE >=10 trades on train, >=3 on test by using looser Fisher thresholds and momentum backup.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_4h_hma_chop_regime_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher(close, period=9):
    """Calculate Ehlers Fisher Transform."""
    close_s = pd.Series(close)
    highest = close_s.rolling(window=period, min_periods=period).max().values
    lowest = close_s.rolling(window=period, min_periods=period).min().values
    range_val = highest - lowest
    range_val = np.where(range_val == 0, 0.001, range_val)
    normalized = (close - lowest) / range_val
    normalized = np.clip(normalized, 0.001, 0.999)
    fisher_input = 0.33 * 2 * (normalized - 0.5) + 0.67 * np.roll(0.33 * 2 * (normalized - 0.5), 1)
    fisher_input[0] = 0.33 * 2 * (normalized[0] - 0.5)
    fisher = 0.5 * np.log((1 + fisher_input) / (1 - fisher_input))
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    range_val = highest - lowest
    range_val = np.where(range_val == 0, 0.001, range_val)
    chop = 100 * np.log10(atr_sum / range_val) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align 4h HMA to 1d (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher(close, 9)
    chop = calculate_choppiness(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Price momentum for backup signals
    momentum_5 = pd.Series(close).pct_change(5).values
    momentum_10 = pd.Series(close).pct_change(10).values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # 4h trend bias (loose filter - just bias, not strict requirement)
        bullish_trend = close[i] > hma_4h_aligned[i]
        bearish_trend = close[i] < hma_4h_aligned[i]
        
        # Fisher Transform signals (LOOSENED thresholds for more trades)
        fisher_long = fisher_prev[i] < -0.8 and fisher[i] >= -0.8
        fisher_short = fisher_prev[i] > 0.8 and fisher[i] <= 0.8
        
        # Choppiness regime filter
        trending_regime = chop[i] < 50  # Below 50 = trending
        ranging_regime = chop[i] >= 50  # Above 50 = ranging
        
        # Adjust position size based on regime
        if ranging_regime:
            size_multiplier = 0.8  # Reduce size in range markets
        else:
            size_multiplier = 1.0
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Fisher crossover
        if fisher_long:
            new_signal = SIZE_ENTRY * size_multiplier
        # Backup: Strong momentum with trend
        elif momentum_5[i] > 0.05 and bullish_trend and rsi[i] < 70:
            new_signal = SIZE_ENTRY * size_multiplier
        # Backup: RSI oversold bounce
        elif rsi[i] < 30 and rsi[i-1] < 35 and bullish_trend:
            new_signal = SIZE_ENTRY * size_multiplier
        
        # === SHORT ENTRY ===
        # Primary: Fisher crossover
        if fisher_short:
            new_signal = -SIZE_ENTRY * size_multiplier
        # Backup: Strong negative momentum with trend
        elif momentum_5[i] < -0.05 and bearish_trend and rsi[i] > 30:
            new_signal = -SIZE_ENTRY * size_multiplier
        # Backup: RSI overbought drop
        elif rsi[i] > 70 and rsi[i-1] > 65 and bearish_trend:
            new_signal = -SIZE_ENTRY * size_multiplier
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF * size_multiplier
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF * size_multiplier
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals