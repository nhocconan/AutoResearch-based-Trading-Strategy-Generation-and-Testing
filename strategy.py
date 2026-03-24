#!/usr/bin/env python3
"""
Experiment #595: 6h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + Volume Confirmation

Hypothesis: 6h timeframe with HMA trend filter + RSI pullback entries provides optimal
balance between trend following and mean reversion. 6h captures multi-day swings without
the noise of lower timeframes. Using 1d/1w HMA for macro bias ensures we trade with
the dominant trend. Volume confirmation filters out weak breakouts.

Key innovations vs failed experiments:
1. HMA slope confirmation (not just price vs HMA) - reduces whipsaw
2. RSI pullback INTO trend (not extreme RSI) - catches continuation, not reversals
3. Volume spike confirmation on entry - validates institutional interest
4. Asymmetric sizing: larger positions when all 3 TF align (1w+1d+6h)
5. ATR-based trailing stop with breakeven trigger

Strategy logic:
1. 1w HMA(21) = macro trend (slowest filter)
2. 1d HMA(21) = medium trend bias
3. 6h HMA(34) = primary trend following
4. 6h RSI(14) pullback to 40-55 (long) or 45-60 (short) in trend
5. Volume > 1.3x 20-bar avg = confirmation
6. ATR(14)*2.5 stoploss, trail to breakeven at 1.5R

Regime-adaptive:
- STRONG TREND (1w+1d+6h all aligned): SIZE=0.30
- MODERATE TREND (1d+6h aligned): SIZE=0.25
- WEAK/TRANSITION: SIZE=0.20 or flat

Target: Sharpe>0.40, trades>=80 train, trades>=10 test, DD>-30%
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_pullback_vol_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA with less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of Volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    hma_6h = calculate_hma(close, period=34)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_WEAK = 0.20
    SIZE_MOD = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    breakeven_triggered = False
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w macro + 1d medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        
        # HMA slope confirmation (5-bar lookback)
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-5] if i >= 5 and not np.isnan(hma_1w_aligned[i-5]) else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-5] if i >= 5 and not np.isnan(hma_1w_aligned[i-5]) else False
        
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 and not np.isnan(hma_1d_aligned[i-5]) else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 and not np.isnan(hma_1d_aligned[i-5]) else False
        
        # === 6h HMA TREND ===
        hma_6h_bull = close[i] > hma_6h[i]
        hma_6h_bear = close[i] < hma_6h[i]
        
        hma_6h_slope_bull = hma_6h[i] > hma_6h[i-5] if i >= 5 and not np.isnan(hma_6h[i-5]) else False
        hma_6h_slope_bear = hma_6h[i] < hma_6h[i-5] if i >= 5 and not np.isnan(hma_6h[i-5]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.3 * vol_sma[i] if vol_sma[i] > 1e-10 else False
        
        # === RSI PULLBACK (not extreme - catching continuation) ===
        # Long: RSI pulled back to 40-55 in uptrend
        rsi_pullback_long = 38.0 <= rsi[i] <= 58.0
        # Short: RSI rallied to 45-62 in downtrend
        rsi_pullback_short = 42.0 <= rsi[i] <= 62.0
        
        # === TREND ALIGNMENT SCORING ===
        # Count how many timeframes align bullish/bearish
        bull_align = sum([htf_bull, hma_1w_slope_bull, hma_1d_slope_bull, hma_6h_bull, hma_6h_slope_bull])
        bear_align = sum([htf_bear, hma_1w_slope_bear, hma_1d_slope_bear, hma_6h_bear, hma_6h_slope_bear])
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        signal_strength = 0
        
        # LONG entries: need bullish alignment + RSI pullback
        if bull_align >= 3 and rsi_pullback_long:
            if bull_align >= 4 and vol_confirmed:
                desired_signal = SIZE_STRONG
                signal_strength = 3
            elif bull_align >= 3:
                desired_signal = SIZE_MOD
                signal_strength = 2
            else:
                desired_signal = SIZE_WEAK
                signal_strength = 1
        
        # SHORT entries: need bearish alignment + RSI pullback
        elif bear_align >= 3 and rsi_pullback_short:
            if bear_align >= 4 and vol_confirmed:
                desired_signal = -SIZE_STRONG
                signal_strength = 3
            elif bear_align >= 3:
                desired_signal = -SIZE_MOD
                signal_strength = 2
            else:
                desired_signal = -SIZE_WEAK
                signal_strength = 1
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trail stop: highest - 2.5*ATR
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            
            # Breakeven trigger at 1.5R
            if not breakeven_triggered and high[i] >= entry_price + 1.5 * 2.5 * entry_atr:
                breakeven_triggered = True
                stop_price = max(stop_price, entry_price + 0.3 * entry_atr)  # small buffer
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trail stop: lowest + 2.5*ATR
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            
            # Breakeven trigger at 1.5R
            if not breakeven_triggered and low[i] <= entry_price - 1.5 * 2.5 * entry_atr:
                breakeven_triggered = True
                stop_price = min(stop_price, entry_price - 0.3 * entry_atr)  # small buffer
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_MOD * 0.9:
            final_signal = SIZE_MOD
        elif desired_signal <= -SIZE_MOD * 0.9:
            final_signal = -SIZE_MOD
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                breakeven_triggered = False
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                breakeven_triggered = False
        
        signals[i] = final_signal
    
    return signals