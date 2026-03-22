#!/usr/bin/env python3
"""
Experiment #086: 30m Supertrend with 4h HMA Trend Filter + Simplified RSI
Hypothesis: 30m Supertrend captures intraday trends. 4h HMA provides appropriate HTF bias
(not 1d which is too slow for 30m). Simplified RSI conditions (single threshold, not range)
to avoid over-filtering that killed #079 (Sharpe=-7.623). Wider ATR stops (3.0x) for 30m noise.

Why this might work on 30m (learning from failures):
- #074 (30m KAMA): -1.600 Sharpe - KAMA too slow for 30m
- #079 (30m RSI + 4h HMA + 1h ST): -7.623 Sharpe - TOO MANY TIMEFRAMES (3 conflicting)
- #080 (30m EMA momentum): -3.202 Sharpe - momentum whipsaw on 30m
- #085 (30m RSI mean reversion): -4.051 Sharpe - pure mean reversion fails on 30m

Key insight: Use ONLY 2 timeframes (30m + 4h), NOT 3. Supertrend proven on 4h (#076 success).
Simplify RSI to single threshold (not range). Wider stops for 30m volatility.

Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
Position sizing: 0.20 base, 0.25 strong signals (conservative for 30m noise).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_4h_hma_rsi_simplified_v1"
timeframe = "30m"
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, trend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = long, -1 = short
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = 1
        else:
            if trend[i-1] == 1:
                if close[i] < lower_band[i-1]:
                    trend[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    trend[i] = 1
                    supertrend[i] = max(upper_band[i], supertrend[i-1])
            else:
                if close[i] > upper_band[i-1]:
                    trend[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    trend[i] = -1
                    supertrend[i] = min(lower_band[i], supertrend[i-1])
    
    return supertrend, trend

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Supertrend
    supertrend, supertrend_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4) - CONSERVATIVE for 30m
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias (appropriate for 30m entries)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === SUPERTREND SIGNAL ===
        supertrend_long = supertrend_direction[i] == 1
        supertrend_short = supertrend_direction[i] == -1
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === SIMPLIFIED RSI FILTER (single threshold, not range) ===
        # Avoid over-filtering that killed #079
        rsi_bullish = rsi[i] > 45  # Simple threshold
        rsi_bearish = rsi[i] < 55  # Simple threshold
        
        # === RSI MOMENTUM ===
        rsi_momentum_long = rsi[i] > 50
        rsi_momentum_short = rsi[i] < 50
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (simplified - 2 paths max) ===
        # Path 1: Supertrend long + 4h bullish + RSI bullish (primary)
        if supertrend_long and bull_trend_4h and rsi_bullish:
            if ema_bullish:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        # Path 2: Supertrend long + EMA bullish (simpler, ensures trades)
        if new_signal == 0.0 and supertrend_long and ema_bullish:
            if rsi_momentum_long:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (simplified - 2 paths max) ===
        # Path 1: Supertrend short + 4h bearish + RSI bearish (primary)
        if supertrend_short and bear_trend_4h and rsi_bearish:
            if ema_bearish:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
        # Path 2: Supertrend short + EMA bearish (simpler, ensures trades)
        if new_signal == 0.0 and supertrend_short and ema_bearish:
            if rsi_momentum_short:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - Wider stops for 30m noise ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 3.0 * ATR below highest close (wider for 30m)
            stoploss_price = highest_close - 3.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 3.0 * ATR above lowest close (wider for 30m)
            stoploss_price = lowest_close + 3.0 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals