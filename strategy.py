#!/usr/bin/env python3
"""
EXPERIMENT #012 - EMA Crossover + RSI Momentum + 1w HMA Trend Filter (1d primary)
=====================================================================================
Hypothesis: Daily timeframe captures major crypto trends while avoiding noise from lower TFs.
Using 1w HMA(21) as the ultimate trend filter ensures we only trade with the macro direction.
EMA(8/21) crossover on 1d provides clean entry signals. RSI(14) > 50 for longs, < 50 for shorts
confirms momentum. ATR(14) stoploss at 2.5*ATR protects against major reversals.

Key features:
- Primary TF: 1d (daily bars = fewer false signals, lower fee churn)
- HTF filter: 1w HMA(21) for macro trend direction
- Entry: EMA(8) crosses EMA(21) with momentum confirmation
- Momentum: RSI(14) > 50 for longs, < 50 for shorts
- Strength: ADX(14) > 20 optional filter (not too strict)
- Stoploss: 2.5*ATR(14) trailing stop
- Position sizing: 0.25-0.35 discrete levels (conservative for daily moves)
- Take profit: Reduce to half at 3R profit, trail stop at 1.5R

Why this should work on 1d:
- Daily bars = ~1500 bars over 4 years = manageable trade frequency
- 1w HMA filter removes counter-trend trades during major reversals
- EMA crossover is proven on daily timeframes for crypto
- Conservative sizing (0.30) protects against 50%+ crypto crashes
- Simple logic = fewer conditions that could block all trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ema_rsi_1whma_1d_v1"
timeframe = "1d"
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


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean().values
    return ema


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
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period - 1, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend filter
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # Base position size (30% of capital)
    MAX_SIZE = 0.35   # Max position size
    MIN_SIZE = 0.25   # Min position size
    HALF_SIZE = 0.15  # Half position for take profit
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    # Track EMA crossover state
    prev_ema_diff = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(ema_fast[i]) or
            np.isnan(ema_slow[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or
            np.isnan(adx[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1w HMA macro trend filter
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        macro_trend = 1 if price_above_1w_hma else -1
        
        # EMA crossover signal
        ema_diff = ema_fast[i] - ema_slow[i]
        ema_crossover_long = (prev_ema_diff <= 0) and (ema_diff > 0)
        ema_crossover_short = (prev_ema_diff >= 0) and (ema_diff < 0)
        prev_ema_diff = ema_diff
        
        # EMA alignment (fast above slow = bullish)
        ema_bullish = ema_fast[i] > ema_slow[i]
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # RSI momentum filter (not too strict)
        rsi_bullish = rsi[i] > 45  # Slightly below 50 to catch early momentum
        rsi_bearish = rsi[i] < 55  # Slightly above 50 to catch early momentum
        
        # ADX strength (optional, not too strict to ensure trades)
        adx_ok = adx[i] > 18  # Lower threshold to allow more trades
        
        # Calculate position size based on ADX strength
        adx_multiplier = min(1.0 + (adx[i] - 18) / 40, 1.15)
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * adx_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: EMA crossover + macro trend up + RSI bullish + ADX ok
        if (ema_crossover_long and macro_trend == 1 and rsi_bullish and adx_ok):
            target_signal = position_size
        
        # Short entry: EMA crossover + macro trend down + RSI bearish + ADX ok
        elif (ema_crossover_short and macro_trend == -1 and rsi_bearish and adx_ok):
            target_signal = -position_size
        
        # Also allow entries on EMA alignment (not just crossover) for more trades
        if target_signal == 0.0 and position_side == 0:
            # Long on alignment if strong momentum
            if (ema_bullish and macro_trend == 1 and rsi[i] > 55 and adx[i] > 22):
                target_signal = position_size
            # Short on alignment if strong momentum
            elif (ema_bearish and macro_trend == -1 and rsi[i] < 45 and adx[i] > 22):
                target_signal = -position_size
        
        # Stoploss and take profit logic
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
                
                # Check take profit (3R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 7.5 * entry_atr:  # 3R profit
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
                    if close[i] <= entry_price - 7.5 * entry_atr:  # 3R profit
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
            # Reduce position to half at 3R profit
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
                # Maintain existing position - check if trend reversed
                ema_reversal_long = ema_bearish  # Fast crossed below slow
                ema_reversal_short = ema_bullish  # Fast crossed above slow
                macro_reversal = (position_side == 1 and macro_trend == -1) or \
                                 (position_side == -1 and macro_trend == 1)
                
                # Exit on EMA reversal OR macro trend reversal
                if ema_reversal_long or ema_reversal_short or macro_reversal:
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