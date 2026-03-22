#!/usr/bin/env python3
"""
Experiment #146: 30m Supertrend + 4h HMA Trend Filter + RSI Pullback + ATR Stop

Hypothesis: #140 (30m Supertrend + 4h HMA + ADX) achieved Sharpe=0.074, proving the concept works.
Adding RSI pullback filter should improve entry timing (enter on dips in uptrend, rallies in downtrend).
Loose RSI conditions (just <50/>50, not extreme values) ensure adequate trade frequency.
Tighter stoploss (2.0 * ATR) reduces drawdown compared to #140.

Why this might beat current best (Sharpe=0.478):
- 30m captures more intraday moves than 4h/12h strategies
- 4h HMA provides stable trend filter (proven in multiple experiments)
- RSI pullback improves win rate without killing trade frequency
- 2.0 ATR stop is tighter than typical 2.5-3.0, reducing DD

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h HMA via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.0 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_4h_hma_rsi_pullback_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator with direction."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)
    supertrend[:] = np.nan
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        
        if i == period:
            supertrend[i] = lower_band[i]
            direction[i] = 1
            continue
        
        if direction[i-1] == 1:
            if close[i] < lower_band[i-1]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            if close[i] > upper_band[i-1]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    return supertrend, direction

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain) * 100, where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (CRITICAL - Rule 2, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(supertrend[i]):
            continue
        if np.isnan(rsi[i]):
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === SUPERTREND SIGNAL ===
        st_bull = st_direction[i] == 1
        st_bear = st_direction[i] == -1
        
        # === RSI PULLBACK FILTER (loose conditions for trade frequency) ===
        # Long: RSI < 55 (not oversold, just not overbought)
        # Short: RSI > 45 (not overbought, just not oversold)
        # This ensures we get trades while still having some filter
        rsi_ok_long = rsi[i] < 55
        rsi_ok_short = rsi[i] > 45
        
        # Strong entry: RSI deeply oversold/overbought
        rsi_strong_long = rsi[i] < 40
        rsi_strong_short = rsi[i] > 60
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # All three must align: 4h bullish + Supertrend bullish + RSI not overbought
        if bull_trend_4h and st_bull and rsi_ok_long:
            if rsi_strong_long:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # All three must align: 4h bearish + Supertrend bearish + RSI not oversold
        if bear_trend_4h and st_bear and rsi_ok_short:
            if rsi_strong_short:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side > 0:
            if highest_close == 0.0:
                highest_close = close[i]
            else:
                highest_close = max(highest_close, close[i])
            stoploss_price = highest_close - 2.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if lowest_close == 0.0:
                lowest_close = close[i]
            else:
                lowest_close = min(lowest_close, close[i])
            stoploss_price = lowest_close + 2.0 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0
        
        # Update position tracking
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals