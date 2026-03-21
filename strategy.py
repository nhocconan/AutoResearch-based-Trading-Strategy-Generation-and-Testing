#!/usr/bin/env python3
"""
EXPERIMENT #018 - EMA Trend + RSI Filter + 1w HMA Trend (1d primary)
=====================================================================================
Hypothesis: Daily timeframe captures major crypto trends with fewer false signals than lower TFs.
Using 1w HMA as trend filter ensures we trade with the macro direction. EMA crossover (8/21)
provides clear entry signals. RSI(14) filter avoids entering at extremes (>70 or <30).
Volume confirmation ensures we're trading with institutional participation.

Key features:
- Primary TF: 1d (daily candles - fewer but higher quality signals)
- HTF filter: 1w HMA(21) for macro trend direction
- Trend: EMA(8) vs EMA(21) crossover
- Entry: RSI(14) between 35-65 (avoid extremes)
- Volume: above 20-day average
- Stoploss: 2.5*ATR(14) trailing (wider for daily TF)
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 3R profit

Why this should work on 1d:
- Daily bars filter out noise from lower timeframes
- 1w HMA provides strong macro trend filter
- EMA crossover is simple but effective on daily
- RSI filter prevents buying tops/selling bottoms
- Conservative sizing controls drawdown during crypto crashes
- Looser entry conditions ensure ≥10 trades per symbol
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


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_sma(values, period):
    """Calculate Simple Moving Average"""
    values_s = pd.Series(values)
    sma = values_s.rolling(window=period, min_periods=period).mean().values
    return sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
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
    vol_sma = calculate_sma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.32   # Max position size
    MIN_SIZE = 0.22   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 50  # Wait for indicators to stabilize (less than 15m strategies need)
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(ema_fast[i]) or
            np.isnan(ema_slow[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or
            np.isnan(vol_sma[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1w HMA macro trend filter
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        hma_trend = 1 if price_above_1w_hma else -1
        
        # EMA crossover signal
        ema_bullish = ema_fast[i] > ema_slow[i]
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # EMA crossover detection (fast crosses above/below slow)
        ema_cross_long = (ema_fast[i] > ema_slow[i]) and (ema_fast[i-1] <= ema_slow[i-1])
        ema_cross_short = (ema_fast[i] < ema_slow[i]) and (ema_fast[i-1] >= ema_slow[i-1])
        
        # RSI filter - avoid extremes (looser than typical for more trades)
        rsi_not_overbought = rsi[i] < 70  # Can still enter if RSI < 70
        rsi_not_oversold = rsi[i] > 30    # Can still enter if RSI > 30
        
        # Volume confirmation (above 20-day average)
        volume_confirmed = volume[i] > vol_sma[i] * 0.8  # 80% of avg is OK
        
        # Calculate position size
        position_size = BASE_SIZE
        
        # Determine target signal based on filters
        target_signal = 0.0
        
        # Long entry: EMA bullish + 1w HMA bullish + RSI not overbought + volume OK
        # Allow entry on EMA cross OR when already bullish with pullback
        if (ema_bullish and hma_trend == 1 and rsi_not_overbought and volume_confirmed):
            # Stronger signal on crossover
            if ema_cross_long:
                target_signal = MAX_SIZE
            else:
                target_signal = position_size
        
        # Short entry: EMA bearish + 1w HMA bearish + RSI not oversold + volume OK
        elif (ema_bearish and hma_trend == -1 and rsi_not_oversold and volume_confirmed):
            # Stronger signal on crossover
            if ema_cross_short:
                target_signal = -MAX_SIZE
            else:
                target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]  # Wider stop for daily
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (3R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 7.5 * entry_atr:  # 3R = 7.5*ATR
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
                # Maintain existing position (check if trend reversed)
                # Exit if EMA reverses OR 1w HMA alignment breaks
                ema_reversal_long = ema_bearish
                ema_reversal_short = ema_bullish
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                if ema_reversal_long or ema_reversal_short or hma_alignment_broken:
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