#!/usr/bin/env python3
"""
Experiment #412: 4h KAMA Adaptive Trend + Daily HMA Bias + RSI Pullback + ATR Stop
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - fast in trends,
slow in chop. Combined with daily HMA for trend bias and RSI pullback entries, this should
generate consistent trades on 4h timeframe while filtering false signals. Key insight from
failures: SIMPLER entry logic = more trades. Using only 2-3 filters instead of 5+.
Timeframe: 4h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
Position size: 0.25 discrete, stoploss 2.5*ATR for 4h timeframe.
Target: Beat Sharpe=0.499 with >=10 trades/symbol on train, >=3 on test.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_daily_hma_rsi_pullback_atr_v1"
timeframe = "4h"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - moves fast in trends, slow in chop.
    ER (Efficiency Ratio) determines smoothing constant.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        price_change = np.abs(close[i] - close[i - period])
        sum_volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_volatility > 0:
            er[i] = price_change / sum_volatility
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
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
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator for trend direction."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if close[i - 1] <= supertrend[i - 1]:
            # Previous trend was bearish or neutral
            if close[i] > upper_band[i]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:
            # Previous trend was bullish
            if close[i] < lower_band[i]:
                supertrend[i] = upper_band[i]
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, trend

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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10)
    rsi = calculate_rsi(close, 14)
    sma50 = calculate_sma(close, 50)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma50[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (long-term direction)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # KAMA trend (adaptive - fast in trends, slow in chop)
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # Supertrend direction
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # RSI pullback levels (relaxed for more trades)
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 65  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 35 and rsi[i] < 65  # Pullback in downtrend
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        # Price vs SMA50
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (simplified - fewer filters = more trades) ===
        # Path 1: KAMA bullish + Daily bullish + RSI momentum (primary)
        if kama_bullish and daily_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Path 2: Supertrend bullish + Daily bullish + RSI ok
        elif st_bullish and daily_bullish and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 3: KAMA bullish + Supertrend bullish + above SMA50
        elif kama_bullish and st_bullish and above_sma50:
            new_signal = SIZE_ENTRY
        # Path 4: Daily bullish + RSI pullback + KAMA turning up
        elif daily_bullish and rsi_pullback_long and kama_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (simplified - fewer filters = more trades) ===
        # Path 1: KAMA bearish + Daily bearish + RSI momentum (primary)
        if kama_bearish and daily_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Path 2: Supertrend bearish + Daily bearish + RSI ok
        elif st_bearish and daily_bearish and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 3: KAMA bearish + Supertrend bearish + below SMA50
        elif kama_bearish and st_bearish and below_sma50:
            new_signal = -SIZE_ENTRY
        # Path 4: Daily bearish + RSI pullback + KAMA turning down
        elif daily_bearish and rsi_pullback_short and kama_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for 4h timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest for 4h timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals