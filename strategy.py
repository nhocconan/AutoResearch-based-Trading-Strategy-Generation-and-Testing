#!/usr/bin/env python3
"""
EXPERIMENT #013 - KAMA Adaptive Trend + BB Regime + 4h Filter (15m primary)
=====================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency,
performing better than static EMAs in crypto's varying volatility regimes.
Combined with Bollinger Band width for regime detection (squeeze = breakout pending)
and 4h KAMA for major trend filter. RSI for entry timing on pullbacks.

Key features:
- Primary TF: 15m (mandatory for this experiment)
- HTF filter: 4h KAMA(21) for major trend direction
- Trend: 15m KAMA(10) with slope confirmation
- Regime: Bollinger Band width percentile (avoid trading in extreme squeeze/expansion)
- Entry: RSI(14) pullback within trend (RSI 35-50 long, 50-65 short)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2.5R profit

Why this should work:
- KAMA adapts to volatility, reducing whipsaws in chop
- BB regime filter avoids entering during extreme conditions
- 4h filter ensures we trade with major trend
- Relaxed RSI conditions (35-50 vs 30-45) generate more trades
- Conservative sizing controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_bbregime_4hfilter_15m_v1"
timeframe = "15m"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    
    return upper, lower, sma, bandwidth


def calculate_kama_slope(kama, lookback=5):
    """Calculate KAMA slope (positive = uptrend, negative = downtrend)"""
    n = len(kama)
    slope = np.zeros(n)
    slope[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(kama[i]) and not np.isnan(kama[i - lookback]):
            slope[i] = (kama[i] - kama[i - lookback]) / kama[i - lookback] * 100
    
    return slope


def calculate_bw_percentile(bandwidth, lookback=100):
    """Calculate Bollinger Band Width percentile rank"""
    n = len(bandwidth)
    bw_pct = np.zeros(n)
    bw_pct[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(bandwidth[i]):
            window = bandwidth[i - lookback:i + 1]
            window = window[~np.isnan(window)]
            if len(window) > 0:
                bw_pct[i] = np.sum(window < bandwidth[i]) / len(window)
    
    return bw_pct


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h KAMA for trend filter
    kama_4h = calculate_kama(df_4h['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate 15m indicators
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slope_15m = calculate_kama_slope(kama_15m, lookback=5)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_pct = calculate_bw_percentile(bb_bandwidth, lookback=100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = 0.14  # Half position for take profit
    
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
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(kama_15m[i]) or
            np.isnan(kama_slope_15m[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or
            np.isnan(bb_pct[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h KAMA trend filter
        price_above_4h_kama = close[i] > kama_4h_aligned[i]
        hma_trend = 1 if price_above_4h_kama else -1
        
        # 15m KAMA trend
        kama_bullish = kama_slope_15m[i] > 0.05  # Slightly positive slope
        kama_bearish = kama_slope_15m[i] < -0.05  # Slightly negative slope
        
        # Price relative to 15m KAMA
        price_above_kama = close[i] > kama_15m[i]
        price_below_kama = close[i] < kama_15m[i]
        
        # Bollinger Band regime filter (avoid extreme squeeze or expansion)
        # Trade when BB width is in normal range (20th-80th percentile)
        bb_normal = 0.20 < bb_pct[i] < 0.80
        
        # RSI pullback conditions (relaxed for more trades)
        rsi_pullback_long = 35 < rsi[i] < 55  # Pullback in uptrend
        rsi_pullback_short = 45 < rsi[i] < 65  # Pullback in downtrend
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h bullish + 15m KAMA bullish + price above KAMA + RSI pullback + BB normal
        if (hma_trend == 1 and kama_bullish and price_above_kama and 
            rsi_pullback_long and bb_normal):
            target_signal = BASE_SIZE
        
        # Short entry: 4h bearish + 15m KAMA bearish + price below KAMA + RSI pullback + BB normal
        elif (hma_trend == -1 and kama_bearish and price_below_kama and 
              rsi_pullback_short and bb_normal):
            target_signal = -BASE_SIZE
        
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
                # Exit if KAMA slope reverses OR 4h KAMA alignment breaks
                kama_reversal_long = kama_slope_15m[i] < -0.05
                kama_reversal_short = kama_slope_15m[i] > 0.05
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
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
                    signals[i] = BASE_SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals