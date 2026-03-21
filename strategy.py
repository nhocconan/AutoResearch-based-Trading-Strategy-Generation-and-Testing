#!/usr/bin/env python3
"""
EXPERIMENT #082 - KAMA Trend + RSI Pullback + BBW Regime + Triple HTF (4h primary)
=====================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts better to crypto volatility
than EMA/HMA, reducing whipsaws in chop. RSI pullback entries (RSI 40-50 in uptrend,
50-60 in downtrend) work better than breakouts on 4h timeframe. Bollinger Band Width
regime filter avoids trading during squeezes (low volatility = chop). Triple HTF
alignment (4h price vs 1d HMA vs 1w HMA) ensures we trade with the major trend.

Key features:
- Primary TF: 4h (mandatory for this experiment)
- HTF filters: 1d HMA(50) + 1w HMA(50) for triple alignment
- Trend: KAMA(10,2,30) - adapts to volatility, less whipsaw than EMA
- Entry: RSI(14) pullback to 40-50 zone in uptrend, 50-60 in downtrend
- Regime: BBW percentile > 40th (avoid squeezes/chop)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete, scaled by KAMA efficiency ratio
- Take profit: Reduce to half at 2R profit, trail stop at 1R

Why this should beat current best (Sharpe=0.490):
- KAMA adapts to volatility better than HMA/EMA (proven in literature)
- Pullback entries have better risk/reward than breakouts on 4h
- BBW regime filter removes 40%+ of trades in chop
- Conservative sizing (0.25-0.30) controls drawdown
- Triple HTF alignment ensures major trend confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_pullback_bbwregime_4h_1d_1w_v1"
timeframe = "4h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - moves fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant (SC)
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(er_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
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
    """Calculate RSI"""
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
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bbw = (upper - lower) / middle
    return upper, lower, middle, bbw


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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    kama, er = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_middle, bbw = calculate_bollinger_bands(close, 20, 2.0)
    
    # Calculate BBW percentile rank (regime filter)
    bbw_pr = calculate_percentile_rank(bbw, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size with strong ER
    MIN_SIZE = 0.20   # Min position size
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
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or
            np.isnan(kama[i]) or np.isnan(er[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bbw_pr[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Triple HTF trend alignment
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # 1d and 1w trend direction
        daily_trend = 1 if price_above_1d_hma else -1
        weekly_trend = 1 if price_above_1w_hma else -1
        
        # KAMA trend direction
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # KAMA Efficiency Ratio (higher = stronger trend, less noise)
        er_strong = er[i] > 0.3  # ER > 0.3 indicates trending market
        
        # BBW regime filter (avoid squeezes/chop)
        bbw_regime = bbw_pr[i] > 0.40  # Only trade when BBW in top 60%
        
        # RSI pullback zones
        rsi_pullback_long = 40 <= rsi[i] <= 55  # Pullback in uptrend
        rsi_pullback_short = 45 <= rsi[i] <= 60  # Pullback in downtrend
        
        # Calculate position size based on ER strength (dynamic sizing)
        er_multiplier = min(1.0 + (er[i] - 0.3) * 2, 1.25)  # Max 1.25x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * er_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + RSI pullback + ER strong + BBW regime + Triple HTF bullish
        if (kama_trend == 1 and rsi_pullback_long and er_strong and bbw_regime and
            daily_trend == 1 and weekly_trend == 1):
            target_signal = position_size
        
        # Short entry: KAMA bearish + RSI pullback + ER strong + BBW regime + Triple HTF bearish
        elif (kama_trend == -1 and rsi_pullback_short and er_strong and bbw_regime and
              daily_trend == -1 and weekly_trend == -1):
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
            signals[i] = HALF_SIZE * np.sign(position_side)
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
                # Exit if KAMA reverses OR HTF alignment breaks OR RSI extremes
                kama_reversal_long = close[i] < kama[i] and position_side == 1
                kama_reversal_short = close[i] > kama[i] and position_side == -1
                hma_alignment_broken = (position_side == 1 and daily_trend == -1) or \
                                       (position_side == -1 and daily_trend == 1)
                rsi_extreme_long = rsi[i] > 70 and position_side == 1  # Overbought exit
                rsi_extreme_short = rsi[i] < 30 and position_side == -1  # Oversold exit
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken or \
                   rsi_extreme_long or rsi_extreme_short:
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