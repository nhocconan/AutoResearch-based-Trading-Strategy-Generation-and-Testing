#!/usr/bin/env python3
"""
Experiment #150: 1d EMA Crossover + 1w HMA Trend Filter + ATR Stop

Hypothesis: Daily timeframe captures major trend moves while weekly HMA provides
stable long-term trend bias. EMA crossover (21/55) is simple enough to generate
adequate trades but filtered by weekly trend to avoid counter-trend positions.
RSI filter prevents entries at extremes. ATR stoploss protects from crashes.

Why 1d might work:
- Fewer trades = less fee drag (critical for profitability)
- Weekly trend filter avoids 2022 crash whipsaw
- Simple EMA crossover ensures trade frequency (learned from #142, #143)
- Daily timeframe less noisy than 15m/1h for trend following

Learning from failures:
- #142, #143: 0 trades due to too many filters
- #145: Negative Sharpe on 15m (too much noise)
- #144: 1d with RSI had Sharpe=0.154 - proves daily can work
- Keep conditions SIMPLE: trend + crossover + RSI range

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_ema_1w_hma_rsi_atr_v1"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0).values
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
    ema_fast = calculate_ema(close, 21)
    ema_slow = calculate_ema(close, 55)
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = higher timeframe trend bias
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === EMA CROSSOVER SIGNAL ===
        # Fast EMA crosses above slow EMA = bullish
        # Fast EMA crosses below slow EMA = bearish
        ema_bull = ema_fast[i] > ema_slow[i]
        ema_bear = ema_fast[i] < ema_slow[i]
        
        # Check for crossover (entry trigger)
        ema_cross_bull = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_bear = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # === RSI FILTER ===
        # Avoid entries at extremes (RSI < 30 or > 70)
        # But allow positions to continue if already in
        rsi_neutral = 35 < rsi[i] < 65
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Weekly bullish + EMA crossover bullish + RSI not overbought
        if bull_trend_1w and ema_cross_bull and rsi[i] < 70:
            new_signal = SIZE_BASE
        
        # Strengthen position if trend is strong
        if bull_trend_1w and ema_bull and rsi_neutral:
            new_signal = SIZE_STRONG
        
        # === SHORT ENTRY CONDITIONS ===
        # Weekly bearish + EMA crossover bearish + RSI not oversold
        if bear_trend_1w and ema_cross_bear and rsi[i] > 30:
            new_signal = -SIZE_BASE
        
        # Strengthen position if trend is strong
        if bear_trend_1w and ema_bear and rsi_neutral:
            new_signal = -SIZE_STRONG
        
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