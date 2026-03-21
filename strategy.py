#!/usr/bin/env python3
"""
EXPERIMENT #014 - Bollinger Squeeze Breakout + KAMA Trend + 4h HMA Filter (30m primary)
=====================================================================================
Hypothesis: Bollinger Band squeeze (low bandwidth) identifies consolidation periods before
volatility expansion. KAMA (Kaufman Adaptive Moving Average) adapts to market noise better
than EMA. Combining squeeze breakout with KAMA trend + 4h HMA filter should capture
high-probability momentum moves while avoiding choppy false breakouts.

Key features:
- Primary TF: 30m
- HTF filter: 4h HMA(21) for major trend direction
- Regime: Bollinger Band Width percentile < 20th (squeeze detection)
- Trend: KAMA(10,2,30) for adaptive trend following
- Entry: Price breaks BB upper/lower during squeeze + volume spike
- Strength: Volume > 1.5x 20-period average
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should beat previous attempts:
- Squeeze filter avoids trading in chop (major cause of previous failures)
- KAMA adapts to volatility better than static EMA/HMA
- Volume confirmation reduces false breakouts
- 4h HMA ensures we trade with higher timeframe trend
- Conservative sizing controls drawdown during crypto crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "bb_squeeze_kama_4hhma_30m_v1"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    change = np.zeros(n)
    volatility = np.zeros(n)
    
    for i in range(er_period, n):
        change[i] = abs(close[i] - close[i - er_period])
        volatility[i] = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
    
    er = np.zeros(n)
    for i in range(er_period, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    for i in range(er_period, n):
        sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] ** 2 * (close[i] - kama[i - 1])
    
    return kama


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    
    return upper, lower, sma, bandwidth


def calculate_volume_spike(volume, period=20):
    """Calculate volume spike indicator"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_std = vol_s.rolling(window=period, min_periods=period).std().values
    
    volume_ratio = volume / vol_avg
    volume_zscore = (volume - vol_avg) / vol_std
    
    return volume_ratio, volume_zscore


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_bandwidth_percentile(bandwidth, lookback=100):
    """Calculate rolling percentile of bandwidth"""
    n = len(bandwidth)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback, n):
        window = bandwidth[i - lookback:i + 1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            percentile[i] = np.sum(bandwidth[i] >= valid_window) / len(valid_window) * 100
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, period=14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    volume_ratio, volume_zscore = calculate_volume_spike(volume, period=20)
    bw_percentile = calculate_bandwidth_percentile(bb_bandwidth, lookback=100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size with strong confirmation
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(atr[i]) or np.isnan(bb_bandwidth[i]) or
            np.isnan(bw_percentile[i]) or np.isnan(volume_ratio[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # KAMA trend direction
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # Bollinger Band squeeze detection (bandwidth in bottom 20th percentile)
        squeeze_active = bw_percentile[i] < 20.0
        
        # Volume spike confirmation
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Breakout detection
        breakout_long = close[i] > bb_upper[i]
        breakout_short = close[i] < bb_lower[i]
        
        # Calculate position size based on volume strength
        volume_multiplier = min(1.0 + (volume_ratio[i] - 1.5) / 3.0, 1.2)
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * volume_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Squeeze + Long breakout + KAMA bullish + 4h HMA bullish + Volume spike
        if (squeeze_active and breakout_long and kama_trend == 1 and 
            hma_trend == 1 and volume_confirmed):
            target_signal = position_size
        
        # Short entry: Squeeze + Short breakout + KAMA bearish + 4h HMA bearish + Volume spike
        elif (squeeze_active and breakout_short and kama_trend == -1 and 
              hma_trend == -1 and volume_confirmed):
            target_signal = -position_size
        
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
                
                # Check take profit (2R from entry, where R = 2*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4*ATR
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
                    if close[i] <= entry_price - 4.0 * entry_atr:  # 2R profit
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
                # Exit if KAMA reverses OR 4h HMA alignment breaks OR squeeze ends without profit
                kama_reversal_long = kama_trend == -1
                kama_reversal_short = kama_trend == 1
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                # Exit if squeeze ends and we're not in profit (failed breakout)
                squeeze_ended = bw_percentile[i] > 50.0
                in_loss = (position_side == 1 and close[i] < entry_price) or \
                          (position_side == -1 and close[i] > entry_price)
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken or \
                   (squeeze_ended and in_loss):
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