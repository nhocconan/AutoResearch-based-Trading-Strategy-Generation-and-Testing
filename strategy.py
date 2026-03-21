#!/usr/bin/env python3
"""
EXPERIMENT #078 - KAMA Trend + RSI Pullback + Weekly Filter (1d primary)
=========================================================================
Hypothesis: Daily timeframe captures major crypto trends with fewer false signals.
KAMA (Kaufman Adaptive Moving Average) adapts to volatility - fast in trends, slow in chop.
RSI pullback to 45-55 zone (not extremes) provides better entry timing than breakouts.
1w HMA filter ensures we trade with the major weekly trend.

Key features:
- Primary TF: 1d (daily bars - fewer but higher quality signals)
- HTF filter: 1w HMA(50) for major trend alignment
- Trend: KAMA(14) with Efficiency Ratio adaptation
- Entry: RSI(14) pullback to 45-55 zone in direction of KAMA trend
- Regime: KAMA slope confirmation + ADX(14) > 20
- Stoploss: 2.5*ATR(14) trailing (wider for daily timeframe)
- Take profit: Reduce to half at 3R profit, trail at 1.5R
- Position sizing: 0.25 base, 0.35 max (discrete levels)

Why this should beat current best (Sharpe=0.490):
- Daily timeframe reduces noise and false signals vs 12h/4h
- KAMA adapts better to crypto volatility regimes than HMA/EMA
- RSI pullback (not breakout) entries have better risk/reward
- Weekly filter prevents trading against major trend
- Conservative sizing controls drawdown during 2022 bear market
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_pullback_weekly_1d_1w_v1"
timeframe = "1d"
leverage = 1.0


def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - moves fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period - 1, n):
        signal = abs(close[i] - close[i - period + 1])
        noise = np.sum(np.abs(np.diff(close[i - period + 1:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama, er


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
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period - 1, n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
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
    
    return adx, plus_di, minus_di


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    kama, er = calculate_kama(close, period=14)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # Calculate KAMA slope (rate of change)
    kama_slope = np.zeros(n)
    for i in range(5, n):
        if kama[i - 5] != 0:
            kama_slope[i] = (kama[i] - kama[i - 5]) / kama[i - 5] * 100
        else:
            kama_slope[i] = 0
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.35   # Max position size with strong signals
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter (major trend direction)
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        weekly_trend = 1 if price_above_1w_hma else -1
        
        # KAMA trend direction
        kama_bullish = close[i] > kama[i] and kama_slope[i] > 0
        kama_bearish = close[i] < kama[i] and kama_slope[i] < 0
        
        # ADX strength filter (trend strength > 20)
        adx_strong = adx[i] > 20
        
        # RSI pullback zone (not overbought/oversold, but pullback)
        rsi_pullback_long = 45 <= rsi[i] <= 55  # Pullback in uptrend
        rsi_pullback_short = 45 <= rsi[i] <= 55  # Pullback in downtrend
        
        # DI confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = minus_di[i] > plus_di[i]
        
        # Calculate position size based on ER (Efficiency Ratio)
        # Higher ER = cleaner trend = larger position
        er_multiplier = min(1.0 + er[i], 1.4)  # Max 1.4x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * er_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + RSI pullback + ADX strong + Weekly bullish + DI+ > DI-
        if (kama_bullish and rsi_pullback_long and adx_strong and 
            weekly_trend == 1 and di_bullish):
            target_signal = position_size
        
        # Short entry: KAMA bearish + RSI pullback + ADX strong + Weekly bearish + DI- > DI+
        elif (kama_bearish and rsi_pullback_short and adx_strong and 
              weekly_trend == -1 and di_bearish):
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
                # Exit if KAMA reverses OR Weekly alignment breaks
                kama_reversal_long = close[i] < kama[i] and kama_slope[i] < 0
                kama_reversal_short = close[i] > kama[i] and kama_slope[i] > 0
                hma_alignment_broken = (position_side == 1 and weekly_trend == -1) or \
                                       (position_side == -1 and weekly_trend == 1)
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken:
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