#!/usr/bin/env python3
"""
Experiment #432: 12h Primary + 1d HTF — KAMA Trend + RSI Pullback + Donchian Confirmation

Hypothesis: 12h timeframe with KAMA adaptive trend + RSI pullback entries can generate
adequate trade frequency (80-200 trades over 4-year train) while maintaining quality.
Key lesson from failed 12h strategies (#422, #426, #427): entry conditions were TOO STRICT.

Changes from failed attempts:
- Use EITHER 1d OR KAMA for bias (not both required)
- Remove Choppiness filter (caused 0 trades in #426, #427)
- Wider RSI thresholds: 25/75 instead of 35/65
- KAMA crossover as primary signal (more frequent than HMA)
- Position hold logic to reduce churn
- Warmup: 200 bars instead of 300

Why this should work:
- KAMA adapts faster than HMA in crypto volatility
- RSI pullback in trend = high probability entries
- 1d HTF bias prevents counter-trend trades
- Simpler logic = more trades (avoid 0-trade failure)

Target: Sharpe > 0.612, 80-200 trades train, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_pullback_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (signal-to-noise ratio).
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (volatility + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    kama_21 = calculate_kama(close, period=10, fast=2, slow=30)
    kama_50 = calculate_kama(close, period=20, fast=2, slow=30)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 12h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(kama_21[i]) or np.isnan(kama_50[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === HTF BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h KAMA) ===
        kama_bullish = kama_21[i] > kama_50[i]
        kama_bearish = kama_21[i] < kama_50[i]
        
        # KAMA crossover detection
        kama_cross_bull = kama_bullish and (i > 0 and not np.isnan(kama_21[i-1]) and 
                                            kama_21[i-1] <= kama_50[i-1])
        kama_cross_bear = kama_bearish and (i > 0 and not np.isnan(kama_21[i-1]) and 
                                            kama_21[i-1] >= kama_50[i-1])
        
        # === RSI SIGNALS (wider thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 30.0  # Mean reversion long
        rsi_overbought = rsi_14[i] > 70.0  # Mean reversion short
        rsi_neutral_long = rsi_14[i] < 50.0  # Pullback in uptrend
        rsi_neutral_short = rsi_14[i] > 50.0  # Pullback in downtrend
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.8:
            position_size = BASE_SIZE * 0.7
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP — Multiple entry methods (LOOSE conditions for more trades)
        # Method 1: KAMA crossover + HTF bullish (primary trend follow)
        if kama_cross_bull and price_above_hma_1d:
            desired_signal = position_size
        # Method 2: RSI pullback in uptrend (KAMA bullish + RSI < 50)
        elif kama_bullish and rsi_neutral_long and price_above_hma_1d:
            desired_signal = position_size * 0.8
        # Method 3: RSI oversold + HTF neutral or bull (mean reversion)
        elif rsi_oversold and (price_above_hma_1d or kama_bullish):
            desired_signal = position_size * 0.7
        # Method 4: Donchian breakout + trend alignment
        elif donchian_breakout_long and (kama_bullish or price_above_hma_1d):
            desired_signal = position_size * 0.6
        
        # SHORT SETUP — Multiple entry methods
        # Method 1: KAMA crossover + HTF bearish
        if kama_cross_bear and price_below_hma_1d:
            desired_signal = -position_size
        # Method 2: RSI pullback in downtrend
        elif kama_bearish and rsi_neutral_short and price_below_hma_1d:
            desired_signal = -position_size * 0.8
        # Method 3: RSI overbought + HTF neutral or bear
        elif rsi_overbought and (price_below_hma_1d or kama_bearish):
            desired_signal = -position_size * 0.7
        # Method 4: Donchian breakdown + trend alignment
        elif donchian_breakout_short and (kama_bearish or price_below_hma_1d):
            desired_signal = -position_size * 0.6
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === RSI EXTREME EXIT ===
        if in_position and position_side > 0 and rsi_14[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 20.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1d and kama_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d and kama_bullish:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (price_above_hma_1d or kama_bullish):
                desired_signal = position_size * 0.5  # Hold with reduced size
            elif position_side < 0 and (price_below_hma_1d or kama_bearish):
                desired_signal = -position_size * 0.5  # Hold with reduced size
        
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