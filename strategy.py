#!/usr/bin/env python3
"""
EXPERIMENT #009 - KAMA Adaptive Trend + 4h HMA Filter + RSI Pullback (1h primary)
=====================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise better than EMA/SMA.
Combining KAMA on 1h with 4h HMA trend filter should capture trends while filtering chop.
RSI pullback entries (RSI < 50 in uptrend, RSI > 50 in downtrend) improve entry timing.
Bollinger Band width regime filter avoids trading during extreme squeeze/expansion.

Key features:
- Primary TF: 1h (mandatory for this experiment)
- HTF filter: 4h HMA(21) for major trend direction
- Trend: KAMA(14) adaptive moving average
- Entry: Price crosses KAMA + RSI confirmation
- Regime: Bollinger Band width percentile (avoid extremes)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should work:
- KAMA adapts to volatility, reducing whipsaws in chop
- 4h HMA filter ensures we trade with higher timeframe trend
- Bollinger regime filter avoids low-opportunity periods
- Conservative sizing (0.25-0.30) controls drawdown
- 1h timeframe generates sufficient trades vs 4h/12h strategies
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_4hhma_rsi_bbregime_1h_v1"
timeframe = "1h"
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


def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    change = np.zeros(n)
    volatility = np.zeros(n)
    
    for i in range(period, n):
        change[i] = abs(close[i] - close[i - period])
        vol_sum = 0.0
        for j in range(i - period + 1, i + 1):
            vol_sum += abs(close[j] - close[j - 1])
        volatility[i] = vol_sum
    
    er = np.zeros(n)
    for i in range(period, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
        sc[i] = sc[i] ** 2  # Square the smoothing constant
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
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
    """Calculate Bollinger Bands and Bandwidth"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    return upper, lower, bandwidth


def calculate_bw_percentile(bandwidth, lookback=100):
    """Calculate Bollinger Bandwidth percentile over lookback period"""
    n = len(bandwidth)
    bw_pct = np.zeros(n)
    bw_pct[:] = np.nan
    
    for i in range(lookback, n):
        window = bandwidth[i - lookback:i + 1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            bw_pct[i] = np.sum(bandwidth[i] >= valid_window) / len(valid_window)
    
    return bw_pct


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
    
    # Calculate 1h indicators
    kama = calculate_kama(close, period=14, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bw_percentile = calculate_bw_percentile(bb_bandwidth, lookback=100)
    
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
    
    min_period = 120  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(bw_percentile[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # KAMA trend direction
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # KAMA slope (compare to 3 bars ago)
        kama_slope = 0
        if i >= 3 and not np.isnan(kama[i - 3]):
            kama_slope = 1 if kama[i] > kama[i - 3] else -1
        
        # RSI confirmation
        rsi_bullish = rsi[i] > 45 and rsi[i] < 70  # Not overbought
        rsi_bearish = rsi[i] < 55 and rsi[i] > 30  # Not oversold
        
        # Bollinger Band regime filter (avoid extremes)
        # Trade when BW percentile is between 0.3 and 0.8 (normal regime)
        bb_regime_ok = 0.25 < bw_percentile[i] < 0.85
        
        # Calculate position size based on regime strength
        if bb_regime_ok:
            position_size = BASE_SIZE
        else:
            position_size = MIN_SIZE  # Reduce size in extreme regimes
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + 4h HMA bullish + KAMA slope up + RSI confirmation + BB regime ok
        if (kama_trend == 1 and hma_trend == 1 and kama_slope == 1 and 
            rsi_bullish and bb_regime_ok):
            # Additional confirmation: price crossed above KAMA recently
            if i >= 2 and close[i - 1] <= kama[i - 1]:
                target_signal = position_size
        
        # Short entry: KAMA bearish + 4h HMA bearish + KAMA slope down + RSI confirmation + BB regime ok
        elif (kama_trend == -1 and hma_trend == -1 and kama_slope == -1 and 
              rsi_bearish and bb_regime_ok):
            # Additional confirmation: price crossed below KAMA recently
            if i >= 2 and close[i - 1] >= kama[i - 1]:
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
                # Exit if KAMA reverses OR 4h HMA alignment breaks
                kama_reversal_long = kama_trend == -1
                kama_reversal_short = kama_trend == 1
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
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals