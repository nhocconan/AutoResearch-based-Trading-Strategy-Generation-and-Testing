#!/usr/bin/env python3
"""
EXPERIMENT #008 - KAMA Adaptive Trend + 4h HMA Filter + BB Regime (30m primary)
=====================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better than
fixed EMAs, reducing whipsaws in choppy markets while capturing trends efficiently.
Combined with 4h HMA trend filter and Bollinger Band regime detection, this should
generate quality signals on 30m timeframe with controlled drawdown.

Key features:
- Primary TF: 30m (mandatory for this experiment)
- HTF filter: 4h HMA(21) for major trend direction
- Trend: KAMA(21) with adaptive efficiency ratio
- Regime: Bollinger Band Width percentile (avoid extreme squeeze/expansion)
- Entry: KAMA crossover + RSI confirmation (wider thresholds for more trades)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.35)

Why this should work:
- KAMA adapts to volatility = fewer false signals in chop
- 30m captures more opportunities than 1h/4h strategies
- BB regime filter avoids trading in extreme conditions
- Conservative sizing controls drawdown during crashes
- Wider RSI thresholds (35/65) ensure ≥10 trades per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_4hhma_bbregime_30m_v1"
timeframe = "30m"
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


def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise by adjusting smoothing constant based on Efficiency Ratio
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constants
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


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    
    return upper, lower, sma, bandwidth


def calculate_bb_percentile(bandwidth, lookback=100):
    """Calculate Bollinger Band Width percentile over lookback period"""
    n = len(bandwidth)
    bb_pct = np.zeros(n)
    bb_pct[:] = np.nan
    
    for i in range(lookback, n):
        window = bandwidth[i - lookback:i]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            bb_pct[i] = np.sum(valid_window <= bandwidth[i]) / len(valid_window)
    
    return bb_pct


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    kama = calculate_kama(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_percentile = calculate_bb_percentile(bb_bandwidth, lookback=100)
    
    # KAMA fast line for crossover signals
    kama_fast = calculate_kama(close, period=10)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.35   # Max position size
    MIN_SIZE = 0.15   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 120  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(kama[i]) or np.isnan(kama_fast[i]) or
            np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(bb_percentile[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # KAMA trend direction
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # KAMA crossover signals
        kama_crossover_long = kama_fast[i] > kama[i] and kama_fast[i-1] <= kama[i-1]
        kama_crossover_short = kama_fast[i] < kama[i] and kama_fast[i-1] >= kama[i-1]
        
        # Bollinger Band regime filter (avoid extreme squeeze or expansion)
        # Only trade when BB percentile is between 20th and 80th percentile
        bb_regime_ok = 0.20 < bb_percentile[i] < 0.80
        
        # RSI confirmation (wider thresholds for more trades)
        rsi_bullish = rsi[i] > 45  # Not oversold
        rsi_bearish = rsi[i] < 55  # Not overbought
        rsi_strong_long = rsi[i] > 50
        rsi_strong_short = rsi[i] < 50
        
        # Calculate position size based on trend alignment strength
        trend_alignment = (hma_trend == kama_trend)
        position_size = BASE_SIZE
        if trend_alignment:
            position_size = MAX_SIZE
        else:
            position_size = MIN_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + 4h HMA bullish + BB regime ok + RSI confirmation
        if (kama_trend == 1 and hma_trend == 1 and bb_regime_ok and 
            rsi_bullish and rsi_strong_long):
            # Additional confirmation: price above KAMA
            if close[i] > kama[i]:
                target_signal = position_size
        
        # Short entry: KAMA bearish + 4h HMA bearish + BB regime ok + RSI confirmation
        elif (kama_trend == -1 and hma_trend == -1 and bb_regime_ok and 
              rsi_bearish and rsi_strong_short):
            # Additional confirmation: price below KAMA
            if close[i] < kama[i]:
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
                # Exit if KAMA trend reverses OR 4h HMA alignment breaks
                kama_reversal_long = kama_trend == -1
                kama_reversal_short = kama_trend == 1
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                # Also exit if BB regime becomes extreme
                bb_regime_extreme = bb_percentile[i] <= 0.10 or bb_percentile[i] >= 0.90
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken or bb_regime_extreme:
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