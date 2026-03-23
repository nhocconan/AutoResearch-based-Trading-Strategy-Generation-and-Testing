#!/usr/bin/env python3
"""
Experiment #391: 4h Primary + 1d HTF — HMA Trend + RSI Pullback + ATR Stop

Hypothesis: Previous strategies over-filtered with CRSI + Donchian + ADX hysteresis.
This strategy SIMPLIFIES entry logic while maintaining quality:

1. HMA(21/63) for trend - faster than KAMA, proven in best strategies
2. RSI(14) pullback entries - more stable than CRSI, triggers more reliably
3. Single HTF (1d HMA) for bias - dual HTF caused 0 trades in #380, #382, #384
4. ADX(14) > 20 for trend confirmation - lower threshold than 25 for more trades
5. ATR(14) trailing stop at 2.5x - protects capital in crash scenarios
6. Position size 0.30 discrete - balances return vs drawdown

Key insight from failures: Strategies with Sharpe=0.000 had TOO STRICT entries.
This strategy loosens entry conditions while keeping HTF bias filter.

Target: 30-50 trades/year on 4h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL).
Must beat current best: mtf_4h_triple_regime_crsi_donchian_1d1w_v1 (Sharpe=0.612)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA, less lag than SMA.
    """
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values using Wilder's smoothing
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20.0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_21 = calculate_hma(close, 21)
    hma_63 = calculate_hma(close, 63)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 4h (target 30-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_21[i]) or np.isnan(hma_63[i]):
            continue
        
        # === HTF BIAS (1d HMA21) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA21 vs HMA63) ===
        hma_bullish = hma_21[i] > hma_63[i]
        hma_bearish = hma_21[i] < hma_63[i]
        
        # === ADX Trend Strength (lower threshold = more trades) ===
        is_trending = adx_14[i] > 20.0
        
        # === RSI Pullback Levels ===
        rsi_oversold = rsi_14[i] < 45.0  # Pullback in uptrend
        rsi_overbought = rsi_14[i] > 55.0  # Rally in downtrend
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP: Need HTF bullish + 4h trend + RSI pullback
        if price_above_hma_1d and hma_bullish:
            if rsi_oversold:
                # Pullback entry in uptrend
                desired_signal = BASE_SIZE
            elif is_trending and hma_21[i] > hma_21[i-1]:
                # Momentum continuation
                desired_signal = BASE_SIZE
        
        # SHORT SETUP: Need HTF bearish + 4h trend + RSI rally
        if price_below_hma_1d and hma_bearish:
            if rsi_overbought:
                # Rally entry in downtrend
                desired_signal = -BASE_SIZE
            elif is_trending and hma_21[i] < hma_21[i-1]:
                # Momentum continuation
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_hma_1d:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d:
            desired_signal = 0.0
        
        # === TREND EXIT (4h HMA crossover against position) ===
        if in_position and position_side > 0 and hma_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_bullish:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1d and hma_bullish:
                desired_signal = BASE_SIZE
            elif position_side < 0 and price_below_hma_1d and hma_bearish:
                desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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