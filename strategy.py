#!/usr/bin/env python3
"""
EXPERIMENT #039 - KAMA Adaptive Trend + RSI Pullback + ADX Filter (1h primary, 4h HTF)
======================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better
than HMA/EMA, reducing whipsaws in choppy conditions. Combined with 4h HMA trend filter,
RSI pullback entries, and ADX strength confirmation, this should reduce false signals
while maintaining good entry timing. Z-score filter avoids extreme mean-reversion traps.

Key features:
- Primary TF: 1h (required for this experiment)
- HTF filter: 4h HMA(21) for major trend direction
- Adaptive trend: KAMA(10) on 1h for volatility-adjusted trend following
- Entry: RSI(14) pullback to 45-55 zone in trend direction
- Strength filter: ADX(14) > 25 (confirms trending market, not chop)
- Regime filter: Z-score(20) between -1.5 and +1.5 (avoid extremes)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels (critical for drawdown control)
- Take profit: Reduce to half at 2R profit

Why this differs from current best:
- Uses KAMA instead of Supertrend (adaptive vs fixed ATR multiplier)
- Adds ADX strength filter (current best uses BB Width)
- Adds Z-score regime filter (avoids extreme mean-reversion setups)
- Primary TF is 1h vs 4h (more signals, need better filters)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_adx_zscore_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise by adjusting smoothing constant based on efficiency ratio
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    sc = np.zeros(n)
    sc[:] = np.nan
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]  # Initialize with SMA
    
    for i in range(period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


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


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    
    # Smooth TR, +DM, -DM using Wilder's method
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized distance from mean)"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / (std + 1e-10)
    return zscore.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators (primary timeframe)
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital - discrete level)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]) or 
            np.isnan(zscore[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter (HTF) - only trade in direction of higher timeframe trend
        hma_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # 1h KAMA trend direction
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # ADX strength filter - only trade when ADX > 25 (trending market)
        adx_strong = adx[i] > 25.0
        
        # RSI pullback zone (45-55 for entry timing in trend direction)
        rsi_pullback_long = 45 <= rsi[i] <= 55
        rsi_pullback_short = 45 <= rsi[i] <= 55
        
        # Z-score regime filter - avoid extreme mean-reversion setups
        zscore_valid = -1.5 <= zscore[i] <= 1.5
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: All filters aligned bullish
        if (hma_trend == 1 and kama_trend == 1 and adx_strong and 
            rsi_pullback_long and zscore_valid):
            target_signal = SIZE
        
        # Short entry: All filters aligned bearish
        elif (hma_trend == -1 and kama_trend == -1 and adx_strong and 
              rsi_pullback_short and zscore_valid):
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * atr[i]:  # 2R = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
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
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                if position_side == 1 and kama_trend == -1:
                    # KAMA trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and kama_trend == 1:
                    # KAMA trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals