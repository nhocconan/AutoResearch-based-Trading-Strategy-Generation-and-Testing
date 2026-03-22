#!/usr/bin/env python3
"""
Experiment #084: 1d Supertrend with 4h HMA Trend Filter + RSI Pullback
Hypothesis: 1d Supertrend captures major trends without whipsaw. 4h HMA provides
responsive trend bias (faster than 1d HMA). RSI pullback entries work better than
breakouts in bear/range markets (2022 crash, 2025 bear).

Why this might work on 1d:
- 1d Supertrend(10,3) filters out noise, only captures sustained moves
- 4h HMA(21) is faster than 1d HMA, provides earlier trend signals
- RSI pullback (30-70 range) ensures entries during trend continuations
- Looser filters than #083 to ensure >=10 trades on train (critical for 1d)
- ATR stoploss at 2.5x protects from catastrophic moves

Timeframe: 1d (REQUIRED for this experiment), HTF: 4h via mtf_data helper.
Position sizing: 0.25 base, 0.30 strong signals (discrete levels per Rule 4).
Stoploss: 2.5 * ATR trailing stop.

Key difference from #083 (12h): Using 1d primary with 4h HTF (not 1d HTF).
4h is more responsive for trend bias while 1d Supertrend filters noise.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_supertrend_4h_hma_rsi_pullback_v1"
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d indicators
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias (faster than 1d)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === SUPERTREND SIGNAL ===
        supertrend_long = supertrend_direction[i] == 1
        supertrend_short = supertrend_direction[i] == -1
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === RSI FILTER (pullback entries, not extremes) ===
        # Looser ranges to ensure trades on 1d timeframe
        rsi_neutral_long = 30 <= rsi[i] <= 70
        rsi_neutral_short = 30 <= rsi[i] <= 70
        
        # RSI momentum
        rsi_momentum_long = rsi[i] > 40
        rsi_momentum_short = rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Path 1: Supertrend long + 4h bullish + RSI neutral (primary)
        if supertrend_long and bull_trend_4h:
            if rsi_neutral_long:
                if ema_bullish:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
        
        # Path 2: Supertrend long + EMA bullish (simpler, ensures trades)
        if supertrend_long and ema_bullish:
            if rsi_momentum_long:
                if new_signal == 0.0:
                    new_signal = SIZE_BASE
        
        # Path 3: Price above 4h HMA + Supertrend long (trend continuation)
        if bull_trend_4h and supertrend_long:
            if close[i] > ema_21[i]:
                if rsi[i] > 35 and rsi[i] < 65:
                    if new_signal == 0.0:
                        new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Path 1: Supertrend short + 4h bearish + RSI neutral (primary)
        if supertrend_short and bear_trend_4h:
            if rsi_neutral_short:
                if ema_bearish:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
        
        # Path 2: Supertrend short + EMA bearish (simpler, ensures trades)
        if supertrend_short and ema_bearish:
            if rsi_momentum_short:
                if new_signal == 0.0:
                    new_signal = -SIZE_BASE
        
        # Path 3: Price below 4h HMA + Supertrend short (trend continuation)
        if bear_trend_4h and supertrend_short:
            if close[i] < ema_21[i]:
                if rsi[i] > 35 and rsi[i] < 65:
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