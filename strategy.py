#!/usr/bin/env python3
"""
EXPERIMENT #048 - HMA Trend + RSI Pullback + Bollinger Regime (1d primary)
=====================================================================================
Hypothesis: On daily timeframe, buying pullbacks in established trends works better
than breakout strategies. HMA(21) defines trend direction, RSI(14) pullback to 40-60
provides entry timing, and Bollinger Band Width regime filter avoids choppy markets.
This differs from #047 by using mean-reversion entries (not breakouts) + BB regime
filter (not ADX) on slower 1d timeframe for fewer but higher-quality trades.

Key features:
- Primary TF: 1d (daily candles - less noise, fewer false signals)
- HTF filter: 1w HMA(50) for major trend alignment
- Trend: HMA(21) slope + price position
- Entry: RSI(14) pullback to 40-60 zone in trending market
- Regime: Bollinger Band Width percentile > 50th (avoid squeeze/chop)
- Stoploss: 2.5*ATR(14) trailing stop
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2.5R profit

Why this should beat #047 (Sharpe=0.490):
- Daily timeframe has fewer whipsaws than 12h
- RSI pullback entries have better risk/reward than breakouts
- BB regime filter removes low-volatility chop periods
- Conservative sizing (0.25-0.30) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_bb_regime_1d_1w_v1"
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
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    return upper, lower, bandwidth


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
    hma_21 = calculate_hma(close, 21)
    hma_48 = calculate_hma(close, 48)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    
    # Calculate BB bandwidth percentile rank (regime filter)
    bb_pr = calculate_percentile_rank(bb_bandwidth, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size
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
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(hma_21[i]) or np.isnan(hma_48[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_bandwidth[i]) or
            np.isnan(bb_pr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter (major trend direction)
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        weekly_trend = 1 if price_above_1w_hma else -1
        
        # Daily trend direction (HMA slope + price position)
        hma_slope = hma_21[i] - hma_21[i - 10] if i >= 10 else 0
        price_above_hma = close[i] > hma_21[i]
        hma_bullish = hma_21[i] > hma_48[i] and hma_slope > 0
        hma_bearish = hma_21[i] < hma_48[i] and hma_slope < 0
        
        # Bollinger Band regime filter (avoid low volatility chop)
        bb_regime_ok = bb_pr[i] > 0.50  # Only trade when bandwidth in top 50th percentile
        
        # RSI pullback zones (entry timing)
        rsi_pullback_long = 40 <= rsi[i] <= 60  # Pullback in uptrend
        rsi_pullback_short = 40 <= rsi[i] <= 60  # Pullback in downtrend
        
        # Calculate position size based on regime strength
        position_size = BASE_SIZE
        if bb_pr[i] > 0.70:  # Strong regime
            position_size = MAX_SIZE
        elif bb_pr[i] < 0.55:  # Weak regime
            position_size = MIN_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Weekly bullish + Daily HMA bullish + RSI pullback + BB regime OK
        if (weekly_trend == 1 and hma_bullish and price_above_hma and
            rsi_pullback_long and bb_regime_ok):
            target_signal = position_size
        
        # Short entry: Weekly bearish + Daily HMA bearish + RSI pullback + BB regime OK
        elif (weekly_trend == -1 and hma_bearish and not price_above_hma and
              rsi_pullback_short and bb_regime_ok):
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
                # Exit if HMA trend reverses OR weekly alignment breaks
                hma_reversal_long = hma_21[i] < hma_48[i] or hma_slope < 0
                hma_reversal_short = hma_21[i] > hma_48[i] or hma_slope > 0
                weekly_alignment_broken = (position_side == 1 and weekly_trend == -1) or \
                                          (position_side == -1 and weekly_trend == 1)
                
                if hma_reversal_long or hma_reversal_short or weekly_alignment_broken:
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