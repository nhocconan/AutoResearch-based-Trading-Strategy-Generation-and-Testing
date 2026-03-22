#!/usr/bin/env python3
"""
Experiment #003: 1h Fisher Transform + 4h KAMA Trend + Choppiness Regime
Hypothesis: Fisher Transform (Ehlers) excels at catching reversals in bear/range markets (2022 crash, 2025 bear).
Combined with 4h KAMA (adaptive, less lag than EMA) for trend bias.
Choppiness Index filters regime: CHOP>61.8 = range (favor reversals), CHOP<38.2 = trend (favor breakout).
This is DIFFERENT from CRSI approach - Fisher normalizes price to Gaussian distribution for cleaner signals.
Entry thresholds LOOSENED to ensure >=10 trades/symbol (learned from exp#001/#002 failures).
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
Position sizing: 0.25-0.30 discrete, stoploss at 2.5*ATR for wider breathing room.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_kama_chop_regime_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    Proven to catch reversals in bear markets better than RSI.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        # Calculate price range
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            continue
        
        # Normalize price to 0-1 range
        price_norm = (close[i] - lowest) / (highest - lowest)
        price_norm = np.clip(price_norm, 0.001, 0.999)  # Avoid log(0)
        
        # Fisher calculation
        fisher_val = 0.5 * np.log((1 + price_norm) / (1 - price_norm))
        
        # Smooth with previous value
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
            fisher_prev[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            fisher_prev[i] = fisher_val
    
    return fisher, fisher_prev

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts to market efficiency.
    ER (Efficiency Ratio) determines smoothing constant.
    Less lag than EMA in trends, less noise in ranges.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy (favor mean reversion)
    CHOP < 38.2 = trending (favor trend following)
    Values between = transitional
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            chop[i] = 50.0
            continue
        
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = high[j] - low[j]
            atr_sum += tr
        
        # CHOP formula
        chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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
    """Calculate RSI with proper min_periods."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    rsi[np.isnan(rsi)] = 50.0
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    kama_4h = calculate_kama(df_4h['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness_index(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - LOOSENED for more trades
        bull_trend = close[i] > kama_4h_aligned[i]
        bear_trend = close[i] < kama_4h_aligned[i]
        
        # Choppiness regime
        range_regime = chop[i] > 55  # LOOSENED from 61.8
        trend_regime = chop[i] < 45  # LOOSENED from 38.2
        
        # Fisher Transform signals - LOOSENED thresholds
        fisher_long_cross = fisher_prev[i] < -1.2 and fisher[i] >= -1.2  # Cross above -1.2
        fisher_short_cross = fisher_prev[i] > 1.2 and fisher[i] <= 1.2   # Cross below +1.2
        
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # RSI confirmation (loose filter)
        rsi_oversold = rsi[i] < 45  # LOOSENED from 30
        rsi_overbought = rsi[i] > 55  # LOOSENED from 70
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i]
        ema_bearish = close[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY === (LOOSENED conditions for more trades)
        # Primary: Fisher long cross + 4h bull trend (works in any regime)
        if fisher_long_cross and bull_trend:
            new_signal = SIZE_MAX
        # Secondary: Fisher oversold + range regime (mean reversion)
        elif fisher_oversold and range_regime:
            new_signal = SIZE_BASE
        # Tertiary: Fisher long cross + RSI not overbought
        elif fisher_long_cross and rsi_oversold:
            new_signal = SIZE_BASE
        # Quaternary: Fisher oversold + EMA bullish (trend pullback)
        elif fisher_oversold and ema_bullish:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY === (LOOSENED conditions for more trades)
        # Primary: Fisher short cross + 4h bear trend (works in any regime)
        if fisher_short_cross and bear_trend:
            new_signal = -SIZE_MAX
        # Secondary: Fisher overbought + range regime (mean reversion)
        elif fisher_overbought and range_regime:
            new_signal = -SIZE_BASE
        # Tertiary: Fisher short cross + RSI not oversold
        elif fisher_short_cross and rsi_overbought:
            new_signal = -SIZE_BASE
        # Quaternary: Fisher overbought + EMA bearish (trend pullback)
        elif fisher_overbought and ema_bearish:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for wider breathing room)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals