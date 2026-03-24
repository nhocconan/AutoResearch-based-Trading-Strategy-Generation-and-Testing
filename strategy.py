#!/usr/bin/env python3
"""
Experiment #131: 6h Primary + 1w/1d HTF — Vol Spike Reversion + BB Mean Revert + ADX Regime

Hypothesis: After 120+ failed experiments, the pattern for 6h is clear:
- CRSI-based strategies consistently fail on 6h (experiments #120, #123, #127 all negative Sharpe)
- Choppiness Index filters are too restrictive and kill trade generation
- SOLUTION: Vol spike reversion (ATR ratio) + Bollinger Band mean reversion + ADX regime
- Vol spikes (ATR(7)/ATR(30) > 1.5) signal panic/reversal points — proven edge in 2022 crash
- BB(20, 2.0) extremes provide mean reversion entries in range markets
- ADX with hysteresis (enter >22, exit <18) avoids whipsaw regime switching
- 1w HMA(50) provides major trend bias without being too restrictive for 6h entries
- LOOSE filters to ensure >=30 trades on train, >=3 on test across ALL symbols

Key design choices:
- Timeframe: 6h (30-60 trades/year target, middle ground between 4h and 12h)
- HTF: 1w HMA(50) for major trend bias
- Entry: Vol spike + BB extreme + ADX regime + RSI confirmation
- Regime: ADX > 22 = trend (breakout), ADX < 18 = range (mean revert), hysteresis 18-22
- Position size: 0.27 (27% of capital, conservative for 6h volatility)
- Stoploss: 2.5x ATR trailing stop
- LOOSE RSI (25-75) and vol spike (1.5) thresholds to ensure trade generation

Target: Sharpe > 0.167 (beat current best), DD > -40%, trades >= 30 train, >= 3 test, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_volspike_bb_adx_hma_1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength regardless of direction
    ADX > 25 = strong trend, ADX < 20 = range/chop
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        else:
            plus_dm[i] = 0.0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
        else:
            minus_dm[i] = 0.0
    
    # Smooth with Wilder's method (EMA with alpha = 1/period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and DX
    plus_di = np.where(atr > 1e-10, 100.0 * plus_di / atr, 0.0)
    minus_di = np.where(atr > 1e-10, 100.0 * minus_di / atr, 0.0)
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    # ADX is smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    adx = calculate_adx(high, low, close, period=14)
    
    # Volatility spike ratio (ATR(7) / ATR(30))
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_ratio = np.zeros(n)
    atr_ratio[:] = np.nan
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE = 0.27  # 27% position size (conservative for 6h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # ADX regime state with hysteresis
    adx_regime = 0  # 0 = neutral, 1 = trend, -1 = range
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === ADX REGIME WITH HYSTERESIS ===
        # Enter trend regime at ADX > 22, exit at ADX < 18
        if adx_regime != 1 and adx[i] > 22.0:
            adx_regime = 1
        elif adx_regime != -1 and adx[i] < 18.0:
            adx_regime = -1
        # Keep current regime if ADX between 18-22 (hysteresis zone)
        
        is_trend = adx_regime == 1
        is_range = adx_regime == -1
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === VOL SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 1.5  # LOOSE threshold for trade generation
        
        # === BOLLINGER BAND POSITION ===
        bb_width = bb_upper[i] - bb_lower[i]
        if bb_width > 1e-10:
            bb_pct = (close[i] - bb_lower[i]) / bb_width
        else:
            bb_pct = 0.5
        
        near_bb_lower = bb_pct < 0.15  # Near lower band
        near_bb_upper = bb_pct > 0.85  # Near upper band
        below_bb_lower = close[i] < bb_lower[i]  # Outside lower band
        above_bb_upper = close[i] > bb_upper[i]  # Outside upper band
        
        # === RSI FILTER (LOOSE for trade generation) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_ok_long = rsi[i] > 25.0
        rsi_ok_short = rsi[i] < 75.0
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_trend:
            # TREND REGIME: Follow breakouts with HTF bias
            if above_bb_upper and htf_bull and rsi_ok_long and hma_bull:
                desired_signal = SIZE
            elif below_bb_lower and htf_bear and rsi_ok_short and hma_bear:
                desired_signal = -SIZE
            # Fallback: strong breakout with HMA confirm
            elif above_bb_upper and hma_bull and rsi[i] > 45.0:
                desired_signal = SIZE * 0.7
            elif below_bb_lower and hma_bear and rsi[i] < 55.0:
                desired_signal = -SIZE * 0.7
                
        elif is_range:
            # RANGE REGIME: Mean revert at BB extremes
            if near_bb_lower and rsi_oversold and not htf_bear:
                desired_signal = SIZE
            elif near_bb_upper and rsi_overbought and not htf_bull:
                desired_signal = -SIZE
            # Fallback: BB extreme with RSI confirm
            elif below_bb_lower and rsi[i] < 35.0:
                desired_signal = SIZE * 0.7
            elif above_bb_upper and rsi[i] > 65.0:
                desired_signal = -SIZE * 0.7
        
        # === VOL SPIKE REVERSION (works in any regime) ===
        # High vol + oversold = long (panic bottom)
        if vol_spike and below_bb_lower and rsi[i] < 30.0:
            desired_signal = SIZE
        # High vol + overbought = short (panic top)
        elif vol_spike and above_bb_upper and rsi[i] > 70.0:
            desired_signal = -SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals