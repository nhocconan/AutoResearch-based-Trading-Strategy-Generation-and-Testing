#!/usr/bin/env python3
"""
EXPERIMENT #079 - KAMA Adaptive Trend + Z-Score Pullback + Dual HTF Filter (15m primary)
========================================================================================
Hypothesis: 15m KAMA (Kaufman Adaptive Moving Average) captures trend direction while
adapting to market noise. Z-score(20) identifies overextended pullbacks within the trend
for optimal entry timing. Dual HTF filter (1h HMA + 4h HMA) ensures we trade with the
major trend. This differs from failed supertrend+rsi strategies by using KAMA's adaptive
nature (better in chop) + Z-score mean reversion entry (better timing than RSI extremes).

Key features:
- Primary TF: 15m (faster entries than 1h/4h strategies)
- HTF filters: 1h HMA(21) + 4h HMA(21) for dual alignment
- Trend: KAMA(10) adaptive moving average
- Entry: Z-score(20) < -1.5 for long pullbacks, > 1.5 for short pullbacks
- Regime: KAMA slope + HTF alignment confirmation
- Stoploss: 2.5*ATR(14) trailing (wider for 15m noise)
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.30)
- Take profit: Reduce to half at 2.5R profit

Why this should beat current best (Sharpe=0.490):
- KAMA adapts to volatility (better than fixed Supertrend in chop)
- Z-score pullback entries (better timing than breakout chasing)
- 15m primary = more opportunities than 12h strategies
- Dual HTF filter = stronger trend confirmation than single HTF
- Conservative sizing (0.25-0.30) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_zscore_pullback_dualhtf_15m_1h_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - moves fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period - 1, n):
        signal = abs(close[i] - close[i - period + 1])
        noise = np.sum(np.abs(np.diff(close[i - period + 1:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    # Calculate KAMA
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] ** 2 * (close[i] - kama[i - 1])
    
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


def calculate_zscore(close, period=20):
    """Calculate rolling Z-score"""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std
    return zscore.values


def calculate_kama_slope(kama, lookback=5):
    """Calculate KAMA slope (rate of change)"""
    n = len(kama)
    slope = np.zeros(n)
    slope[:] = np.nan
    
    for i in range(lookback, n):
        if kama[i - lookback] != 0:
            slope[i] = (kama[i] - kama[i - lookback]) / kama[i - lookback] * 100
    
    return slope


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    kama = calculate_kama(close, period=10)
    kama_slope = calculate_kama_slope(kama, lookback=5)
    atr = calculate_atr(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size with strong confirmation
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]) or
            np.isnan(kama[i]) or np.isnan(kama_slope[i]) or
            np.isnan(atr[i]) or np.isnan(zscore[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Dual HTF trend alignment
        price_above_1h_hma = close[i] > hma_1h_aligned[i]
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        
        # HTF trend direction
        hourly_trend = 1 if price_above_1h_hma else -1
        four_hour_trend = 1 if price_above_4h_hma else -1
        
        # KAMA trend direction
        kama_bullish = close[i] > kama[i] and kama_slope[i] > 0
        kama_bearish = close[i] < kama[i] and kama_slope[i] < 0
        
        # Z-score pullback signals (mean reversion within trend)
        # For long: price pulled back (zscore < -1.0) but trend is bullish
        # For short: price rallied (zscore > 1.0) but trend is bearish
        zscore_oversold = zscore[i] < -1.0
        zscore_overbought = zscore[i] > 1.0
        
        # Strong Z-score for entry trigger
        zscore_entry_long = zscore[i] < -1.5
        zscore_entry_short = zscore[i] > 1.5
        
        # Calculate position size based on HTF alignment strength
        htf_alignment_score = 0
        if hourly_trend == 1 and four_hour_trend == 1:
            htf_alignment_score = 2  # Strong bullish
        elif hourly_trend == -1 and four_hour_trend == -1:
            htf_alignment_score = -2  # Strong bearish
        elif hourly_trend == four_hour_trend:
            htf_alignment_score = hourly_trend  # Weak alignment
        else:
            htf_alignment_score = 0  # Conflicted
        
        # Determine target signal based on all filters
        target_signal = 0.0
        position_size = BASE_SIZE
        
        if htf_alignment_score == 2:
            position_size = MAX_SIZE
            # Long entry: KAMA bullish + HTF aligned bullish + Z-score pullback
            if kama_bullish and zscore_entry_long:
                target_signal = position_size
        elif htf_alignment_score == -2:
            position_size = MAX_SIZE
            # Short entry: KAMA bearish + HTF aligned bearish + Z-score pullback
            if kama_bearish and zscore_entry_short:
                target_signal = -position_size
        elif htf_alignment_score == 1:
            position_size = BASE_SIZE
            # Weak long signal
            if kama_bullish and zscore_entry_long:
                target_signal = position_size
        elif htf_alignment_score == -1:
            position_size = BASE_SIZE
            # Weak short signal
            if kama_bearish and zscore_entry_short:
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
                
                # Check take profit (2.5R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 6.25 * entry_atr:  # 2.5R = 6.25*ATR
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
                    if close[i] <= entry_price - 6.25 * entry_atr:  # 2.5R profit
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
            # Reduce position to half at 2.5R profit
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
                # Exit if KAMA reverses OR HTF alignment breaks
                kama_reversal_long = close[i] < kama[i] and kama_slope[i] < 0
                kama_reversal_short = close[i] > kama[i] and kama_slope[i] > 0
                hma_alignment_broken = (position_side == 1 and four_hour_trend == -1) or \
                                       (position_side == -1 and four_hour_trend == 1)
                
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