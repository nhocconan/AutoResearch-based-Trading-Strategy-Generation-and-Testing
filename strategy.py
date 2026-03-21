#!/usr/bin/env python3
"""
EXPERIMENT #028 - KAMA Adaptive Trend + 1d HMA Filter + RSI Pullback (4h primary)
=====================================================================================
Hypothesis: 4h timeframe captures multi-day trends while avoiding noise of lower TFs.
KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - fast in trends,
slow in chop. Combined with 1d HMA(21) for major trend direction, this filters out
counter-trend trades. RSI(14) pullback entries (RSI < 50 long, RSI > 50 short) are
less strict than previous attempts to ensure we generate sufficient trades.

Key features:
- Primary TF: 4h (captures 2-5 day holds)
- HTF filter: 1d HMA(21) for major trend direction
- Trend: KAMA(10, 2, 30) adaptive moving average
- Entry: Price pullback to KAMA + RSI confirmation
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should work:
- 4h is less noisy than 15m/1h, more signals than 12h/1d
- KAMA adapts to volatility, reducing whipsaws in chop
- 1d HMA filter ensures we trade with weekly/monthly trend
- Looser RSI thresholds (40/60 instead of 45/55) generate more trades
- Conservative sizing controls drawdown during crypto crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_1dhma_4h_v2"
timeframe = "4h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - moves fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 0:
            er[i] = signal / noise
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


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    # Use EMA for smoothing (Wilder's method)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if np.isnan(avg_loss[i]) or avg_loss[i] == 0:
            rsi[i] = 100.0
        elif np.isnan(avg_gain[i]):
            rsi[i] = 0.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=max(1, period // 2)).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=max(1, int(np.sqrt(period)))).mean()
    return hma.values


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bands)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend filter
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 50  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(donchian_upper[i]) or
            atr[i] == 0 or atr[i] != atr[i]):  # Second check catches NaN
            signals[i] = 0.0
            continue
        
        # 1d HMA trend filter (major trend direction)
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        hma_trend = 1 if price_above_1d_hma else -1
        
        # KAMA trend (adaptive)
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # RSI conditions (looser thresholds for more trades)
        rsi_oversold = rsi[i] < 50  # Less strict than 45
        rsi_overbought = rsi[i] > 50  # Less strict than 55
        
        # Donchian breakout confirmation
        donchian_breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # Calculate position size (dynamic based on volatility)
        atr_pct = atr[i] / close[i] if close[i] > 0 else 0.02
        vol_adjustment = 0.02 / max(atr_pct, 0.005)  # Normalize to 2% ATR
        vol_adjustment = np.clip(vol_adjustment, 0.8, 1.2)
        position_size = np.clip(BASE_SIZE * vol_adjustment, MIN_SIZE, MAX_SIZE)
        
        # Determine target signal based on filters
        target_signal = 0.0
        
        # Long entry: 1d HMA bullish + KAMA bullish + RSI not overbought
        # Less strict: only need 2 of 3 conditions for entry
        long_conditions = [hma_trend == 1, kama_trend == 1, rsi_oversold]
        long_score = sum(long_conditions)
        
        if long_score >= 2 and rsi[i] > 30:  # RSI > 30 avoids crash entries
            target_signal = position_size
        
        # Short entry: 1d HMA bearish + KAMA bearish + RSI not oversold
        short_conditions = [hma_trend == -1, kama_trend == -1, rsi_overbought]
        short_score = sum(short_conditions)
        
        if short_score >= 2 and rsi[i] < 70:  # RSI < 70 avoids FOMO entries
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if KAMA reverses OR 1d HMA alignment breaks strongly
                kama_reversal_long = kama_trend == -1
                kama_reversal_short = kama_trend == 1
                hma_strong_reversal = (position_side == 1 and hma_trend == -1 and close[i] < kama[i]) or \
                                      (position_side == -1 and hma_trend == 1 and close[i] > kama[i])
                
                if kama_reversal_long or kama_reversal_short or hma_strong_reversal:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals