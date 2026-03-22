#!/usr/bin/env python3
"""
Experiment #129: 1h KAMA Adaptive Trend + 4h HMA Filter + RSI Pullback + ATR Stop

Hypothesis: Adapting the BEST performing strategy (mtf_4h_kama_1d_hma_adx_atr_v1 
with Sharpe=0.478) for 1h timeframe with key modifications:
- 1h KAMA(21) for adaptive trend following (proven in winning 4h strategy)
- 4h HMA(21) for HTF trend bias (better ratio than 1d for 1h primary)
- RSI(14) pullback entries (enter on dips in uptrend, rallies in downtrend)
- ATR(14) trailing stop at 2.5*ATR for capital protection
- Discrete position sizing 0.25-0.35 to limit drawdown

Why 1h might work better than 12h:
- More trade opportunities (12h had very few trades)
- Faster reaction to trend changes
- RSI pullback entries work better on lower timeframes
- Still reduces noise compared to 5m/15m strategies that failed

Key improvements over failed 1h strategies:
- No mean-reversion as primary (RSI -1.7 Sharpe failed)
- No Fisher Transform (failed multiple times)
- No Choppiness Index (failed)
- Simple trend + pullback + stoploss (proven formula)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_rsi_pullback_atr_v1"
timeframe = "1h"
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

def calculate_kama(close, period=21, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    More responsive in trends, smoother in chop.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow:
        return kama
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing Constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 21)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
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
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === KAMA ADAPTIVE TREND ===
        # Price above KAMA = bullish momentum
        bull_kama = close[i] > kama[i]
        bear_kama = close[i] < kama[i]
        
        # KAMA slope (momentum confirmation)
        kama_slope = kama[i] - kama[i-5] if i >= 5 else 0
        kama_bull_slope = kama_slope > 0
        kama_bear_slope = kama_slope < 0
        
        # === RSI PULLBACK ENTRY ===
        # In uptrend: enter on RSI pullback to 40-50 zone
        # In downtrend: enter on RSI rally to 50-60 zone
        rsi_pullback_long = rsi[i] < 55  # Not overbought, room to run
        rsi_pullback_short = rsi[i] > 45  # Not oversold, room to drop
        
        # RSI extreme for strong entries
        rsi_strong_long = rsi[i] < 45  # Deep pullback
        rsi_strong_short = rsi[i] > 55  # Strong rally
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: 4h bullish + KAMA bullish + KAMA slope up + RSI pullback
        if bull_trend_4h and bull_kama and kama_bull_slope and rsi_strong_long:
            new_signal = SIZE_STRONG
        # Moderate: 4h bullish + KAMA bullish + RSI not overbought
        elif bull_trend_4h and bull_kama and rsi_pullback_long:
            new_signal = SIZE_BASE
        # Weak (ensure trades): 4h bullish + KAMA bullish
        elif bull_trend_4h and bull_kama:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: 4h bearish + KAMA bearish + KAMA slope down + RSI rally
        if bear_trend_4h and bear_kama and kama_bear_slope and rsi_strong_short:
            new_signal = -SIZE_STRONG
        # Moderate: 4h bearish + KAMA bearish + RSI not oversold
        elif bear_trend_4h and bear_kama and rsi_pullback_short:
            new_signal = -SIZE_BASE
        # Weak (ensure trades): 4h bearish + KAMA bearish
        elif bear_trend_4h and bear_kama:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
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