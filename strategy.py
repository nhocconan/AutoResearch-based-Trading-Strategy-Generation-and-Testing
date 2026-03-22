#!/usr/bin/env python3
"""
Experiment #083: 12h Supertrend with 1d HMA Trend Filter + RSI Pullback
Hypothesis: 12h Supertrend captures sustained trends without whipsaw. 1d HMA provides
higher-timeframe bias. RSI pullback entries (not breakouts) work better in bear/range markets.
Key insight from #076 success: Supertrend + 1d HMA + RSI worked on 4h. Adapting for 12h
with simpler entry conditions to ensure sufficient trades on all symbols.

Why this might work on 12h:
- Supertrend(10,3) is proven trend indicator that works on slower timeframes
- 1d HMA(21) filters counter-trend trades (critical for BTC/ETH in 2022 crash)
- RSI pullback (not breakout) entries work better in bear/range markets
- Fewer filters = more trades (learned from #077 negative Sharpe due to over-filtering)
- ATR stoploss at 2.5x protects from catastrophic moves

Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.30 strong signals (discrete levels per Rule 4).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_1d_hma_rsi_pullback_v2"
timeframe = "12h"
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
        if np.isnan(atr[i]):
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Supertrend
    supertrend, supertrend_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === SUPERTREND SIGNAL ===
        supertrend_long = supertrend_direction[i] == 1
        supertrend_short = supertrend_direction[i] == -1
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === RSI FILTER (pullback entries, not extremes) ===
        # For longs: RSI pulled back but still bullish (40-60 range)
        rsi_pullback_long = 35 <= rsi[i] <= 60
        # For shorts: RSI bounced but still bearish (40-65 range)
        rsi_pullback_short = 40 <= rsi[i] <= 65
        
        # RSI momentum
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Path 1: Supertrend long + 1d bullish + RSI pullback (primary)
        if supertrend_long and bull_trend_1d:
            if rsi_pullback_long or rsi_momentum_long:
                if ema_bullish:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
        
        # Path 2: Supertrend long + EMA bullish (simpler, ensures trades)
        if supertrend_long and ema_bullish:
            if rsi[i] > 40 and rsi[i] < 70:
                if new_signal == 0.0:
                    new_signal = SIZE_BASE
        
        # Path 3: Price above 1d HMA + Supertrend long (trend continuation)
        if bull_trend_1d and supertrend_long:
            if close[i] > ema_21[i]:
                if rsi[i] > 45 and rsi[i] < 65:
                    if new_signal == 0.0:
                        new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Path 1: Supertrend short + 1d bearish + RSI pullback (primary)
        if supertrend_short and bear_trend_1d:
            if rsi_pullback_short or rsi_momentum_short:
                if ema_bearish:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
        
        # Path 2: Supertrend short + EMA bearish (simpler, ensures trades)
        if supertrend_short and ema_bearish:
            if rsi[i] > 30 and rsi[i] < 60:
                if new_signal == 0.0:
                    new_signal = -SIZE_BASE
        
        # Path 3: Price below 1d HMA + Supertrend short (trend continuation)
        if bear_trend_1d and supertrend_short:
            if close[i] < ema_21[i]:
                if rsi[i] > 35 and rsi[i] < 55:
                    if new_signal == 0.0:
                        new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            stoploss_price = entry_price - 2.5 * atr[i] if position_side > 0 else entry_price + 2.5 * atr[i]
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            stoploss_price = entry_price - 2.5 * atr[i] if position_side > 0 else entry_price + 2.5 * atr[i]
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            stoploss_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals