#!/usr/bin/env python3
"""
Experiment #449: 12h KAMA-RSI Trend Following with Daily HMA Filter

Hypothesis: After 431 failed experiments, the key insight for 12h timeframe is:
1. 12h sits between 4h (noisy) and 1d (too slow) - needs balanced approach
2. Previous 12h failures (#437, #443) had either 0 trades or negative Sharpe
3. KAMA (Kaufman Adaptive Moving Average) adapts to volatility - better than EMA
4. RSI thresholds must be LOOSE (25/75) to ensure sufficient trade frequency
5. 1d HMA filter provides trend bias without being too slow (unlike 1w)

Strategy Components:
1. KAMA(10,50) crossover - adaptive trend following (ER-based smoothing)
2. RSI(14) with 25/75 thresholds - looser for more trades on 12h
3. 1d HMA(21) trend bias - via mtf_data helper (call ONCE before loop)
4. ATR(14) trailing stop at 2.5x - protects against 2022-style crashes
5. Position size: 0.30 discrete - balances return vs drawdown

Why this should work on 12h:
- KAMA adapts to market regime (trending vs ranging)
- Looser RSI thresholds ensure >10 trades/year per symbol
- 1d HMA filter prevents counter-trend disasters
- Simpler than failed ensemble strategies (#437, #443)
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_daily_hma_trend_atr_v1"
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

def calculate_kama(close, fast_period=10, slow_period=50):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    SC (Smoothing Constant) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    close_s = pd.Series(close)
    change = np.abs(close_s.diff())
    signal = np.abs(close_s.diff().values)
    
    # Sum of absolute changes over slow_period
    noise = change.rolling(window=slow_period, min_periods=slow_period).sum().values
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(slow_period, n):
        if noise[i] > 1e-10:
            er[i] = signal[i] / noise[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # KAMA calculation
    kama[slow_period - 1] = close[slow_period - 1]
    for i in range(slow_period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
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
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    kama_fast = calculate_kama(close, fast_period=10, slow_period=50)
    kama_slow = calculate_kama(close, fast_period=20, slow_period=100)
    rsi = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA CROSSOVER SIGNAL ===
        kama_bull = kama_fast[i] > kama_slow[i]
        kama_bear = kama_fast[i] < kama_slow[i]
        
        # === RSI ENTRY TIMING (loose thresholds for more trades) ===
        rsi_oversold = rsi[i] < 35  # Looser than 30 for more long entries
        rsi_overbought = rsi[i] > 65  # Looser than 70 for more short entries
        
        # === SMA FILTER (avoid extreme counter-trend) ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: KAMA bull + RSI oversold + daily trend bull OR above SMA50
        if kama_bull and bull_trend_1d:
            if rsi_oversold or above_sma50:
                new_signal = SIZE
        
        # SHORT ENTRY: KAMA bear + RSI overbought + daily trend bear OR below SMA50
        if kama_bear and bear_trend_1d:
            if rsi_overbought or below_sma50:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals