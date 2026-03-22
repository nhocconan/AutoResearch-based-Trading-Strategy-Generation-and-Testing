#!/usr/bin/env python3
"""
Experiment #078: 1d KAMA + Fisher Transform with 1w HMA Trend Filter
Hypothesis: 1d timeframe is ideal for adaptive trend-following with clear entry signals.
KAMA (Kaufman Adaptive Moving Average) reduces whipsaw in ranging markets while following trends.
Fisher Transform provides clear entry signals at extremes (catches reversals in bear markets).
1w HMA provides strong trend bias without killing trade frequency on 1d.
Key insight: Previous 1d strategies failed due to regime-switching complexity or mean-reversion focus.
Simple adaptive trend + Fisher entries + weekly trend filter = robust across bull/bear/range.
This strategy uses:
- KAMA(10, ER=10) for adaptive trend following (less lag than EMA, less whipsaw than SMA)
- Fisher Transform(9) for entry timing (long when crosses above -1.5, short when crosses below +1.5)
- 1w HMA(21) for weekly trend bias (long only above, short only below)
- ATR(14) trailing stop at 2.5x for risk management
- Discrete position sizing (0.25-0.30 levels)
Why this might work: KAMA adapts to market efficiency ratio, performing well in both trending and ranging.
Fisher Transform normalizes price to Gaussian distribution, providing clear reversal signals.
1w HMA ensures we trade with the dominant weekly trend. 1d has fewer false signals than lower TFs.
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_fisher_1w_hma_adaptive_v2"
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

def calculate_kama(close, period=10, er_period=10):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise using Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = prior_KAMA + SC * (price - prior_KAMA)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period - 1, n):
        net_change = np.abs(close[i] - close[i - er_period + 1])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period + 1:i + 1])))
        if sum_changes > 0:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (2 + 1)  # fast = 2 periods
    slow_sc = 2.0 / (2 + 30)  # slow = 30 periods
    
    # Initialize KAMA
    kama[er_period - 1] = close[er_period - 1]
    
    for i in range(er_period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian-like distribution for clearer signals.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = normalized price
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    # Normalize price to -1 to +1 range
    for i in range(period - 1, n):
        highest = np.max(close[i - period + 1:i + 1])
        lowest = np.min(close[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        # Normalize to 0-1, then scale to -0.99 to +0.99
        x = 2.0 * (close[i] - lowest) / (highest - lowest) - 1.0
        x = np.clip(x, -0.99, 0.99)  # prevent division by zero in log
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Signal line (1-period lag)
        if i > 0:
            fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 10, 10)
    fisher, fisher_signal = calculate_fisher(close, 9)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = weekly trend bias (strong filter)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long = False
        if i > 1 and not np.isnan(fisher_signal[i]):
            fisher_long = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short = False
        if i > 1 and not np.isnan(fisher_signal[i]):
            fisher_short = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # Fisher extreme levels (alternative entry)
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === RSI FILTER (avoid extreme entries) ===
        rsi_ok_long = rsi[i] < 70  # not overbought
        rsi_ok_short = rsi[i] > 30  # not oversold
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Path 1: Fisher reversal + weekly trend bullish + KAMA bullish
        if fisher_long and bull_trend_1w and kama_bullish:
            if rsi_ok_long:
                new_signal = SIZE_STRONG
        
        # Path 2: Fisher oversold + weekly trend bullish (mean reversion in uptrend)
        if fisher_oversold and bull_trend_1w:
            if kama_bullish or rsi[i] < 50:
                new_signal = SIZE_BASE
        
        # Path 3: KAMA bullish + weekly trend bullish + RSI momentum (trend continuation)
        if kama_bullish and bull_trend_1w:
            if 40 <= rsi[i] <= 65:
                new_signal = SIZE_BASE
        
        # Path 4: Simple KAMA cross with weekly confirmation (ensure trades happen)
        if kama_bullish and bull_trend_1w:
            if rsi[i] > 35 and rsi[i] < 75:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Path 1: Fisher reversal + weekly trend bearish + KAMA bearish
        if fisher_short and bear_trend_1w and kama_bearish:
            if rsi_ok_short:
                new_signal = -SIZE_STRONG
        
        # Path 2: Fisher overbought + weekly trend bearish (mean reversion in downtrend)
        if fisher_overbought and bear_trend_1w:
            if kama_bearish or rsi[i] > 50:
                new_signal = -SIZE_BASE
        
        # Path 3: KAMA bearish + weekly trend bearish + RSI momentum (trend continuation)
        if kama_bearish and bear_trend_1w:
            if 35 <= rsi[i] <= 60:
                new_signal = -SIZE_BASE
        
        # Path 4: Simple KAMA cross with weekly confirmation (ensure trades happen)
        if kama_bearish and bear_trend_1w:
            if rsi[i] > 25 and rsi[i] < 65:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals