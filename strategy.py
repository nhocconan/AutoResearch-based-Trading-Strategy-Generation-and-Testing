#!/usr/bin/env python3
"""
Experiment #1104: 4h Primary + 1d HTF — Fisher Transform Reversal with Trend Filter

Hypothesis: After 800+ failed experiments, key insight: simple trend-following fails
in bear/range markets (2022 crash, 2025 test period). Fisher Transform excels at
catching reversals in mean-reverting regimes while still capturing trend moves.

Why this should beat Sharpe=0.612:
1. Fisher Transform (period=9) catches reversals at extremes (-1.5/+1.5 levels)
2. 1d HMA provides macro trend filter without over-complication
3. ADX hysteresis (enter >22, hold >18) reduces whipsaw exits
4. Loose Fisher thresholds ensure 30-50 trades/year on 4h
5. Position size 0.30 with 2.5x ATR trailing stop controls drawdown

Timeframe: 4h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 30-50 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_reversal_1d_hma_adx_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    diff = 2 * wma1 - wma2
    sqrt_period = max(1, int(np.sqrt(period)))
    hma = wma(diff, sqrt_period)
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    Excellent for identifying reversal points in mean-reverting markets.
    
    Formula:
    1. Price = (0.33 * (2*((H+L)/2 - L1)/(H1-L1)) + 0.67 * Price1)
    2. Restrict Price to -0.99 to +0.99
    3. Fisher = 0.5 * ln((1+Price)/(1-Price))
    4. Trigger = Fisher shifted by 1
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, trigger
    
    price = np.zeros(n)
    price[0] = 0.0
    
    for i in range(1, n):
        hl2 = (high[i] + low[i]) / 2.0
        if i == 1:
            hl2_prev = hl2
            range_val = high[i] - low[i]
            if range_val < 1e-10:
                range_val = 1e-10
            price[i] = 0.0
        else:
            range_val = high[i-1] - low[i-1]
            if range_val < 1e-10:
                range_val = 1e-10
            price[i] = 0.33 * (2.0 * (hl2 - low[i-1]) / range_val - 1.0) + 0.67 * price[i-1]
        
        # Restrict to -0.99 to +0.99 to avoid ln domain errors
        price[i] = np.clip(price[i], -0.99, 0.99)
        
        # Fisher transform
        if abs(price[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1.0 + price[i]) / (1.0 - price[i]))
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
    
    # Trigger is Fisher shifted by 1
    trigger[1:] = fisher[:-1]
    trigger[0] = fisher[0]
    
    return fisher, trigger

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = weak/choppy market.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_s > 1e-10
    plus_di[mask] = 100.0 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100.0 * minus_dm_s[mask] / tr_s[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_sma(close, period=200):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    rsi_4h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    sma200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            continue
        if np.isnan(rsi_4h[i]) or np.isnan(atr[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma200[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === LONG TERM TREND (SMA200) ===
        above_sma200 = close[i] > sma200[i]
        below_sma200 = close[i] < sma200[i]
        
        # === TREND STRENGTH (ADX with hysteresis) ===
        # Enter when ADX > 22, hold when ADX > 18
        adx_strong_entry = adx[i] > 22.0
        adx_strong_hold = adx[i] > 18.0
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        
        # === RSI CONFIRMATION ===
        rsi_neutral = 35.0 < rsi_4h[i] < 65.0
        rsi_oversold = rsi_4h[i] < 45.0
        rsi_overbought = rsi_4h[i] > 55.0
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY ===
        # Fisher reversal + macro bull + ADX strong + RSI confirmation
        if fisher_cross_up or (fisher_oversold and rsi_oversold):
            if macro_bull and above_sma200 and adx_strong_entry:
                desired_signal = current_size
        
        # === SHORT ENTRY ===
        # Fisher reversal + macro bear + ADX strong + RSI confirmation
        if fisher_cross_down or (fisher_overbought and rsi_overbought):
            if macro_bear and below_sma200 and adx_strong_entry:
                desired_signal = -current_size
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bull and ADX still strong enough
                if macro_bull and adx_strong_hold:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if macro still bear and ADX still strong enough
                if macro_bear and adx_strong_hold:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses or Fisher overbought
            if macro_bear or fisher[i] > 2.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses or Fisher oversold
            if macro_bull or fisher[i] < -2.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals