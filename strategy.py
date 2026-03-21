#!/usr/bin/env python3
"""
EXPERIMENT #072 - KAMA Trend + RSI Pullback + Weekly Filter + BB Regime (1d primary)
=====================================================================================
Hypothesis: Daily timeframe captures major crypto trends with fewer false signals than
lower timeframes. KAMA (Kaufman Adaptive MA) adapts to volatility better than EMA/HMA.
Weekly HMA filter ensures we only trade with the major trend. Bollinger Band width
regime filter avoids choppy sideways markets. RSI pullback entries improve risk/reward.

Key features:
- Primary TF: 1d (daily candles - slower, higher quality signals)
- HTF filter: 1w HMA(50) for major trend alignment
- Trend: KAMA(10,2,30) adaptive moving average
- Entry: RSI(14) pullback to 40-60 zone in trending market
- Regime: Bollinger Band width > 20th percentile (avoid extreme squeeze/expansion)
- Stoploss: 2.5*ATR(14) trailing (wider for daily timeframe)
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2.5R profit

Why this should beat current best (Sharpe=0.490):
- Daily timeframe = fewer but higher quality trades (less fee churn)
- KAMA adapts to volatility better than static MAs
- Weekly trend filter prevents counter-trend trades
- BB regime filter avoids chop (major cause of losses in prior strategies)
- Conservative sizing (0.25-0.30) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_bbregime_weekly_1d_1w_v1"
timeframe = "1d"
leverage = 1.0


def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(efficiency_period, n):
        signal = abs(close[i] - close[i - efficiency_period])
        noise = 0.0
        for j in range(i - efficiency_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    sc = np.zeros(n)
    sc[:] = np.nan
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(efficiency_period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    # Initialize KAMA with SMA of first efficiency_period bars
    kama[efficiency_period] = np.mean(close[:efficiency_period + 1])
    
    for i in range(efficiency_period + 1, n):
        if not np.isnan(sc[i]):
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
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bandwidth = np.zeros(n)
    
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    # Bandwidth = (Upper - Lower) / Middle
    for i in range(period - 1, n):
        if middle[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bandwidth[i] = 0
    
    return upper, lower, middle, bandwidth


def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank"""
    n = len(series)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(series[i]):
            window_data = series[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                pr[i] = np.sum(window_data <= series[i]) / len(window_data)
    
    return pr


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
    kama = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_middle, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    
    # Calculate BB bandwidth percentile rank (regime filter)
    bb_bw_pr = calculate_percentile_rank(bb_bandwidth, 100)
    
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
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_bandwidth[i]) or
            np.isnan(bb_bw_pr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter (major trend direction)
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        weekly_trend = 1 if price_above_1w_hma else -1
        
        # KAMA trend direction (price vs KAMA)
        price_above_kama = close[i] > kama[i]
        kama_trend = 1 if price_above_kama else -1
        
        # KAMA slope (trend strength)
        kama_slope = 0
        if i >= 5 and not np.isnan(kama[i - 5]):
            kama_slope = 1 if kama[i] > kama[i - 5] else -1
        
        # Bollinger Band regime filter (avoid extreme squeeze or expansion)
        # Only trade when bandwidth is in normal range (20th-80th percentile)
        bb_regime_ok = 0.20 <= bb_bw_pr[i] <= 0.80
        
        # RSI pullback zone (not overbought/oversold for entry)
        rsi_pullback_long = 40 <= rsi[i] <= 60  # Pullback in uptrend
        rsi_pullback_short = 40 <= rsi[i] <= 60  # Pullback in downtrend
        
        # RSI momentum confirmation
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Calculate position size based on trend strength
        trend_strength = 0
        if weekly_trend == 1 and kama_trend == 1 and kama_slope == 1:
            trend_strength = 3  # Strong bullish
        elif weekly_trend == -1 and kama_trend == -1 and kama_slope == -1:
            trend_strength = 3  # Strong bearish
        elif weekly_trend == kama_trend:
            trend_strength = 2  # Moderate
        else:
            trend_strength = 1  # Weak
        
        position_size = MIN_SIZE + (trend_strength - 1) * (MAX_SIZE - MIN_SIZE) / 2
        position_size = min(MAX_SIZE, max(MIN_SIZE, position_size))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Weekly bullish + KAMA bullish + RSI pullback + BB regime OK
        if (weekly_trend == 1 and kama_trend == 1 and kama_slope == 1 and
            rsi_pullback_long and rsi_bullish and bb_regime_ok):
            target_signal = position_size
        
        # Short entry: Weekly bearish + KAMA bearish + RSI pullback + BB regime OK
        elif (weekly_trend == -1 and kama_trend == -1 and kama_slope == -1 and
              rsi_pullback_short and rsi_bearish and bb_regime_ok):
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
                # Exit if KAMA reverses OR weekly trend breaks
                kama_reversal_long = close[i] < kama[i] and kama_trend == -1
                kama_reversal_short = close[i] > kama[i] and kama_trend == 1
                weekly_alignment_broken = (position_side == 1 and weekly_trend == -1) or \
                                          (position_side == -1 and weekly_trend == 1)
                
                if kama_reversal_long or kama_reversal_short or weekly_alignment_broken:
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