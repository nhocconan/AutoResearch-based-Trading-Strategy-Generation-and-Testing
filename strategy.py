#!/usr/bin/env python3
"""
Experiment #669: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback + Donchian

Hypothesis: Previous strategies failed due to overly complex regime detection and
too-strict entry thresholds. This strategy uses SIMPLER logic:
1. 1d HMA for macro trend bias (only trade with HTF trend)
2. 4h HMA for primary trend direction
3. RSI pullback entries (not extremes) — RSI 35-45 for long, 55-65 for short
4. Donchian(20) breakout confirmation
5. ATR(14) trailing stop at 2.5x
6. LOOSE thresholds to ensure 30+ trades/year

Why this should work:
- 4h TF = ~25-40 trades/year (sweet spot for fee drag vs signal quality)
- 1d HMA filter prevents counter-trend trades (major failure mode)
- RSI pullback (not extremes) ensures entries happen during retracements
- Donchian breakout confirms momentum
- Simple logic = fewer conditions that can all fail simultaneously
- Position size 0.25-0.30 to limit drawdown during 2022 crash

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_donchian_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi_raw = 100 - (100 / (1 + rs))
        rsi[period:] = rsi_raw[period-1:]
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    n = len(high)
    donchian_upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return donchian_upper, donchian_lower

def calculate_keltner(high, low, close, atr_period=10, atr_mult=2.0):
    """Keltner Channel for volatility-based bands."""
    n = len(close)
    atr = calculate_atr(high, low, close, atr_period)
    
    ema = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    kc_upper = ema + atr_mult * atr
    kc_lower = ema - atr_mult * atr
    
    return kc_upper, kc_lower, ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=10)
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    kc_upper, kc_lower, kc_mid = calculate_keltner(high, low, close, atr_period=10, atr_mult=2.0)
    
    # Calculate and align HTF indicators (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # RSI on 1d for additional filter
    rsi_1d_raw = calculate_rsi(df_1d['close'].values, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.28
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(hma_4h[i]) or np.isnan(rsi_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === HTF TREND BIAS (1d HMA + RSI) ===
        htf_bullish = close[i] > hma_1d_aligned[i] and rsi_1d_aligned[i] > 45
        htf_bearish = close[i] < hma_1d_aligned[i] and rsi_1d_aligned[i] < 55
        
        # === 4h TREND (HMA) ===
        hma_bullish = close[i] > hma_4h[i] and hma_4h_fast[i] > hma_4h[i]
        hma_bearish = close[i] < hma_4h[i] and hma_4h_fast[i] < hma_4h[i]
        
        # === RSI PULLBACK (LOOSE thresholds for trade generation) ===
        # Long: RSI pulled back to 35-50 zone in uptrend
        rsi_pullback_long = 35 <= rsi_4h[i] <= 50
        # Short: RSI rallied to 50-65 zone in downtrend
        rsi_pullback_short = 50 <= rsi_4h[i] <= 65
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === KELTNER POSITION ===
        kc_near_lower = close[i] < kc_lower[i] if not np.isnan(kc_lower[i]) else False
        kc_near_upper = close[i] > kc_upper[i] if not np.isnan(kc_upper[i]) else False
        
        # === VOLATILITY CHECK ===
        atr_ratio = atr_4h[i] / (np.nanmean(atr_4h[max(0,i-50):i]) + 1e-10) if i >= 50 else 1.0
        vol_normal = 0.5 < atr_ratio < 2.0  # Not extreme volatility
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Condition 1: HTF bullish + 4h HMA bullish + RSI pullback
        if htf_bullish and hma_bullish and rsi_pullback_long and vol_normal:
            desired_signal = SIZE_LONG
        # Condition 2: HTF bullish + Donchian breakout + RSI not overbought
        elif htf_bullish and donchian_breakout_long and rsi_4h[i] < 70:
            desired_signal = SIZE_LONG
        # Condition 3: Price at Keltner lower + HTF not bearish + RSI oversold
        elif kc_near_lower and not htf_bearish and rsi_4h[i] < 40:
            desired_signal = SIZE_LONG
        
        # === SHORT ENTRY ===
        # Condition 1: HTF bearish + 4h HMA bearish + RSI pullback
        elif htf_bearish and hma_bearish and rsi_pullback_short and vol_normal:
            desired_signal = -SIZE_SHORT
        # Condition 2: HTF bearish + Donchian breakdown + RSI not oversold
        elif htf_bearish and donchian_breakout_short and rsi_4h[i] > 30:
            desired_signal = -SIZE_SHORT
        # Condition 3: Price at Keltner upper + HTF not bullish + RSI overbought
        elif kc_near_upper and not htf_bullish and rsi_4h[i] > 60:
            desired_signal = -SIZE_SHORT
        
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HMA still bullish AND RSI not extremely overbought
                if hma_bullish and rsi_4h[i] < 75:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HMA still bearish AND RSI not extremely oversold
                if hma_bearish and rsi_4h[i] > 25:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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