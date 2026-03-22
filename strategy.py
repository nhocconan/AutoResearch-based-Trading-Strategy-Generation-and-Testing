#!/usr/bin/env python3
"""
Experiment #149: 12h RSI Mean Reversion + 1d HMA Trend Filter + ATR Stop

Hypothesis: 12h timeframe is ideal for capturing multi-day swings while avoiding
noise of lower TFs. 1d HMA provides stable trend bias. RSI mean reversion works
well in 2025 bear/range market (learned from research). Simple conditions ensure
adequate trade frequency (≥10 trades per symbol) - critical after #142, #143 
failures with 0 trades.

Why 12h might work:
- Slower than 1h/4h, captures bigger moves with less whipsaw
- More trades than 1d strategies (like #144 with Sharpe=0.154)
- RSI mean reversion proven in bear/range markets (2022 crash, 2025 test)
- 1d HMA trend filter avoids counter-trend disasters

Learning from failures:
- #142, #143: 0 trades from too many filters - keep conditions SIMPLE
- #140: Sharpe=0.074 proved Supertrend+HTF concept works
- #145: 15m too noisy, negative Sharpe
- Current best: Sharpe=0.478 on 4h - 12h should compete with less noise

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi_1d_hma_meanrev_atr_v1"
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
    """Calculate RSI using standard Wilder's method."""
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI Streak component for Connors RSI.
    Measures consecutive up/down days.
    """
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert to RSI-like scale (0-100)
    streak_rsi = np.zeros(n)
    for i in range(period, n):
        pos_count = np.sum(streak[i-period+1:i+1] > 0)
        streak_rsi[i] = (pos_count / period) * 100
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component for Connors RSI.
    Measures where current return ranks vs past N periods.
    """
    n = len(close)
    pr = np.zeros(n)
    
    for i in range(period, n):
        returns = np.diff(close[i-period+1:i+1])
        current_return = returns[-1] if len(returns) > 0 else 0
        rank = np.sum(returns[:-1] < current_return) / max(1, len(returns) - 1)
        pr[i] = rank * 100
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Better for mean reversion than standard RSI.
    """
    rsi_fast = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_fast + streak_rsi + pr) / 3.0
    return crsi

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
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Also calculate fast RSI for quicker signals
    rsi_fast = calculate_rsi(close, 7)
    
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
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME TREND BIAS ===
        # 1d HMA = trend direction
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === MEAN REVERSION SIGNALS ===
        # Connors RSI extremes for mean reversion entries
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        
        # Standard RSI extremes (looser for more trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Fast RSI for quicker entries
        rsi_fast_oversold = rsi_fast[i] < 35
        rsi_fast_overbought = rsi_fast[i] > 65
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Condition 1: 1d bullish + CRSI oversold (trend + mean reversion)
        if bull_trend_1d and crsi_oversold:
            new_signal = SIZE_STRONG
        # Condition 2: 1d bullish + RSI oversold (simpler, more trades)
        elif bull_trend_1d and rsi_oversold:
            new_signal = SIZE_BASE
        # Condition 3: RSI very oversold regardless of trend (catch bottoms)
        elif rsi[i] < 30 or crsi[i] < 15:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Condition 1: 1d bearish + CRSI overbought (trend + mean reversion)
        if bear_trend_1d and crsi_overbought:
            new_signal = -SIZE_STRONG
        # Condition 2: 1d bearish + RSI overbought (simpler, more trades)
        elif bear_trend_1d and rsi_overbought:
            new_signal = -SIZE_BASE
        # Condition 3: RSI very overbought regardless of trend (catch tops)
        elif rsi[i] > 70 or crsi[i] > 85:
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