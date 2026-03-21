#!/usr/bin/env python3
"""
EXPERIMENT #036 - KAMA Trend + Weekly HMA Filter + RSI Pullback (1d primary, 1w HTF)
====================================================================================
Hypothesis: Daily timeframe captures major crypto trends with less noise than intraday.
KAMA (Kaufman Adaptive MA) adapts to market efficiency - fast in trends, slow in chop.
Weekly HMA(50) provides major trend alignment filter. RSI pullback to 45-55 zone entries
reduce false breakouts. Bollinger Band Width regime filter avoids low-volatility chop.
This differs from previous attempts by using daily as primary (slower, more reliable)
with weekly HTF filter instead of daily filter on 4h data.

Key features:
- Primary TF: 1d (daily candles)
- HTF filter: 1w HMA(50) for major trend direction
- Trend: KAMA(10, 2, 30) adaptive moving average
- Entry: RSI(14) pullback to 45-55 zone in trend direction
- Regime: Bollinger Band Width > 40th percentile (trending market only)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit, trail stop at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_weekly_hma_rsi_1d_1w_v1"
timeframe = "1d"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        if not np.isnan(close[i]):
            price_change = abs(close[i] - close[i - er_period])
            volatility = 0.0
            for j in range(i - er_period + 1, i + 1):
                if not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                    volatility += abs(close[j] - close[j - 1])
            if volatility > 0:
                er[i] = price_change / volatility
            else:
                er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA with SMA
    kama[er_period] = np.nanmean(close[:er_period + 1])
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        if np.isnan(er[i]) or np.isnan(kama[i - 1]):
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        if not np.isnan(close[i]):
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    band_width = (upper - lower) / sma
    return upper.values, lower.values, band_width.values


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
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    kama = calculate_kama(close, 10, 2, 30)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2)
    
    # Calculate Bollinger Band Width percentile rank (regime filter)
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 150  # Wait for all indicators to stabilize (KAMA needs ~40, HMA needs ~50, BB PR needs 100)
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_width_pr[i]) or 
            atr[i] == 0 or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter (HTF)
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        
        # Daily KAMA trend direction
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # KAMA slope (additional confirmation)
        kama_slope = 0
        if i >= 5 and not np.isnan(kama[i - 5]):
            kama_slope = 1 if kama[i] > kama[i - 5] else -1
        
        # Regime filter: only trade when BB Width is in top 60% (trending market)
        regime_valid = bb_width_pr[i] > 0.60
        
        # RSI pullback zone (45-55 for entry timing in strong trends)
        rsi_pullback_long = 45 <= rsi[i] <= 55
        rsi_pullback_short = 45 <= rsi[i] <= 55
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 50
        rsi_momentum_short = rsi[i] < 50
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + Weekly trend bullish + RSI pullback + Regime valid + RSI momentum
        if (kama_trend == 1 and weekly_trend == 1 and kama_slope == 1 and 
            rsi_pullback_long and regime_valid and rsi_momentum_long):
            target_signal = SIZE
        
        # Short entry: KAMA bearish + Weekly trend bearish + RSI pullback + Regime valid + RSI momentum
        elif (kama_trend == -1 and weekly_trend == -1 and kama_slope == -1 and 
              rsi_pullback_short and regime_valid and rsi_momentum_short):
            target_signal = -SIZE
        
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
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[i]:  # 2R = 5*ATR
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
                    if close[i] <= entry_price - 5.0 * atr[i]:  # 2R profit
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
                    # Trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and kama_trend == 1:
                    # Trend reversed, exit short
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