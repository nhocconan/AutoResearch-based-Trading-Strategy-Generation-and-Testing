#!/usr/bin/env python3
"""
EXPERIMENT #044 - Multi-Filter Trend Pullback (30m primary, 4h/1d HTF)
================================================================================
Hypothesis: 30m timeframe captures swing moves better than 15m (less noise) but
faster than 1h (more opportunities). Combining 4h HMA trend + 1d HMA major trend
+ ADX strength filter + RSI pullback zone creates high-probability entries only
when all conditions align. Bollinger Band Width regime filter avoids chop.

Key features:
- Primary TF: 30m (this experiment)
- HTF filters: 4h HMA(21) for intermediate trend, 1d HMA(50) for major trend
- Entry: RSI(14) pullback to 45-55 zone (not extremes - avoids fakeouts)
- Strength: ADX(14) > 25 (only trade when trend has momentum)
- Regime: BB Width > 40th percentile (avoid squeeze/chop)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels (conservative)
- Take profit: Reduce to half at 2R, trail stop at 1R

Why this differs from failed attempts:
- ADX filter reduces trades in weak trends (major cause of DD in #043)
- RSI 45-55 zone (not 30/70) captures pullbacks without waiting for extremes
- Dual HTF alignment (4h + 1d) ensures we trade with both intermediate and major trend
- Conservative size (0.25-0.30) limits drawdown during inevitable losing streaks
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "multi_filter_trend_pullback_30m_4h_1d_v1"
timeframe = "30m"
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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    # Calculate True Range and Directional Movement
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
    
    # Smooth using Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    di_minus = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx, di_plus, di_minus


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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - handles shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx, di_plus, di_minus = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2)
    
    # Calculate Bollinger Band Width percentile rank (regime filter)
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28  # Entry position size (28% of capital)
    SIZE_HOLD = 0.25   # Hold size (slightly reduced after entry)
    HALF_SIZE = SIZE_ENTRY / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 1.0
    profit_target_hit = False
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]) or 
            np.isnan(bb_width_pr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # === TREND FILTERS (HTF) ===
        # 4h HMA trend
        trend_4h = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # 1d HMA major trend
        trend_1d = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # === STRENGTH FILTER ===
        # ADX must be > 25 (trending market, not chop)
        adx_valid = adx[i] > 25
        
        # DI+ > DI- for long bias, DI- > DI+ for short bias
        di_long_bias = di_plus[i] > di_minus[i]
        di_short_bias = di_minus[i] > di_plus[i]
        
        # === REGIME FILTER ===
        # BB Width must be > 40th percentile (avoid squeeze/chop)
        regime_valid = bb_width_pr[i] > 0.40
        
        # === ENTRY FILTER (RSI pullback zone) ===
        # RSI 45-55 zone (pullback, not extreme)
        rsi_pullback_long = 45 <= rsi[i] <= 55
        rsi_pullback_short = 45 <= rsi[i] <= 55
        
        # === DETERMINE TARGET SIGNAL ===
        target_signal = 0.0
        
        # Long entry: All filters align
        if (trend_4h == 1 and trend_1d == 1 and  # Both HTF trends bullish
            adx_valid and di_long_bias and        # Strong trend, DI+ dominant
            regime_valid and                       # Not in squeeze
            rsi_pullback_long):                    # Pullback entry
            target_signal = SIZE_ENTRY
        
        # Short entry: All filters align
        elif (trend_4h == -1 and trend_1d == -1 and  # Both HTF trends bearish
              adx_valid and di_short_bias and         # Strong trend, DI- dominant
              regime_valid and                         # Not in squeeze
              rsi_pullback_short):                     # Pullback entry
            target_signal = -SIZE_ENTRY
        
        # === STOPLOSS AND TAKE PROFIT LOGIC ===
        stoploss_triggered = False
        take_profit_triggered = False
        trend_reversal_exit = False
        
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
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR
                        take_profit_triggered = True
                
                # Check trend reversal (4h trend flipped)
                if trend_4h == -1:
                    trend_reversal_exit = True
                    
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
                
                # Check trend reversal (4h trend flipped)
                if trend_4h == 1:
                    trend_reversal_exit = True
        
        # === APPLY SIGNAL ===
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 1.0
            profit_target_hit = False
            
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
            
        elif trend_reversal_exit:
            # Exit on trend reversal
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 1.0
            profit_target_hit = False
            
        else:
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
                # Maintain existing position
                if profit_target_hit:
                    signals[i] = HALF_SIZE * position_side
                else:
                    signals[i] = SIZE_HOLD * position_side
            else:
                signals[i] = 0.0
    
    return signals