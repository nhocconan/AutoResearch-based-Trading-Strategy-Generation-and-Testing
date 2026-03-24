#!/usr/bin/env python3
"""
Experiment #657: 15m Primary + 4h/12h HTF — Dual Regime Adaptive Strategy

Hypothesis: 15m timeframe with REGIME-ADAPTIVE logic should work in both trending (2021)
and ranging/choppy (2022-2024) markets. Previous 15m strategies failed with 0 trades because
entry conditions were too strict. This uses Choppiness Index to DETECT regime, then applies
different logic:

1. RANGE REGIME (CHOP > 61.8): Mean reversion at Bollinger Bands + RSI extremes
   - Long: price < BB_lower + RSI(7) < 30 + 4h HMA bullish bias
   - Short: price > BB_upper + RSI(7) > 70 + 4h HMA bearish bias
   - This generates trades in 2022-2024 choppy period

2. TREND REGIME (CHOP < 38.2): Trend following with HMA + Donchian breakout
   - Long: price > 15m HMA + 4h HMA bullish + Donchian(20) breakout
   - Short: price < 15m HMA + 4h HMA bearish + Donchian(20) breakdown
   - This captures 2021 bull run and any 2025+ trends

3. TRANSITION (38.2 <= CHOP <= 61.8): Reduced size, wait for confirmation

Key innovations:
- Regime detection adapts to market conditions (proven in academic literature)
- LOOSE entry thresholds to ensure >=50 trades/year on 15m
- 4h HMA for direction filter (HTF bias)
- 12h Choppiness for regime confirmation (avoid false signals)
- Position size: 0.15-0.20 (smaller for 15m frequency control)
- ATR(14) trailing stop at 2.5x for risk management

Target: Sharpe>0.40, trades>=50 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_adaptive_chop_bb_rsi_4h12h_v1"
timeframe = "15m"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = rangebound, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh - ll > 1e-10 and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands - mean reversion levels"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma * 100.0  # bandwidth percentage
    
    return upper, lower, width

def calculate_rsi(close, period=7):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0.0)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_hma(close, period):
    """Hull Moving Average - smooth trend indicator"""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    chop_12h_raw = calculate_choppiness(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        period=14
    )
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h_raw)
    
    # Calculate 15m indicators
    chop_15m = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    rsi_7 = calculate_rsi(close, period=7)
    hma_15m = calculate_hma(close, period=21)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing for 15m (smaller due to higher frequency)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_15m[i]) or np.isnan(bb_upper[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        # Use 12h CHOP for regime confirmation (more stable than 15m)
        chop_12h = chop_12h_aligned[i]
        chop_15m_val = chop_15m[i]
        
        # Regime: range, trend, or transition
        if not np.isnan(chop_12h):
            is_range = chop_12h > 61.8
            is_trend = chop_12h < 38.2
        else:
            # Fallback to 15m chop if 12h not available
            is_range = chop_15m_val > 61.8
            is_trend = chop_15m_val < 38.2
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m TREND DIRECTION ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # === ENTRY LOGIC - LOOSE CONDITIONS TO ENSURE TRADES ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion at BB extremes
        if is_range:
            # Long: price at BB lower + RSI oversold + HTF bull bias (preferred)
            if close[i] <= bb_lower[i] and rsi_7[i] < 35:
                if htf_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE  # weaker signal without HTF confirmation
            
            # Short: price at BB upper + RSI overbought + HTF bear bias (preferred)
            elif close[i] >= bb_upper[i] and rsi_7[i] > 65:
                if htf_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # TREND REGIME: Trend following with breakout
        elif is_trend:
            # Long: HTF bull + 15m HMA bull + price above Donchian upper
            if htf_bull and hma_15m_bull and close[i] >= donchian_upper[i]:
                desired_signal = SIZE_STRONG
            # Long: HTF bull + 15m HMA bull (weaker, no breakout required)
            elif htf_bull and hma_15m_bull and close[i] > hma_15m[i]:
                desired_signal = SIZE_BASE
            
            # Short: HTF bear + 15m HMA bear + price below Donchian lower
            elif htf_bear and hma_15m_bear and close[i] <= donchian_lower[i]:
                desired_signal = -SIZE_STRONG
            # Short: HTF bear + 15m HMA bear (weaker, no breakdown required)
            elif htf_bear and hma_15m_bear and close[i] < hma_15m[i]:
                desired_signal = -SIZE_BASE
        
        # TRANSITION REGIME: Reduced size, only strong signals
        else:
            # Only trade if BOTH HTF and 15m agree strongly
            if htf_bull and hma_15m_bull and close[i] >= donchian_upper[i]:
                desired_signal = SIZE_BASE * 0.5
            elif htf_bear and hma_15m_bear and close[i] <= donchian_lower[i]:
                desired_signal = -SIZE_BASE * 0.5
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
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
        
        signals[i] = final_signal
    
    return signals