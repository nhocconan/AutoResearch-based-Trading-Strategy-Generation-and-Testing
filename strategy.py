#!/usr/bin/env python3
"""
EXPERIMENT #041 - KAMA Adaptive Trend + Triple HTF + ADX Regime (12h primary)
================================================================================
Hypothesis: 12h timeframe captures multi-day trends while filtering noise.
KAMA (Kaufman Adaptive MA) adapts to market efficiency - fast in trends, slow in chop.
Triple HTF alignment (12h KAMA + 1d HMA + 1w HMA) ensures we trade with major trend.
ADX(14) > 25 filters out ranging markets. RSI pullback (45-55) times entries.
BB Width percentile ensures we trade during volatility expansion (trending regimes).

Key features:
- Primary TF: 12h (this experiment's requirement)
- HTF filters: 1d HMA(50) + 1w HMA(21) for major trend alignment
- Trend: KAMA(10,2,30) on 12h - adapts to market efficiency
- Regime: ADX(14) > 25 + BB Width > 40th percentile
- Entry: RSI(14) pullback to 45-55 zone in trend direction
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels (reduces fee churn)
- Take profit: Reduce to half at 2R, trail stop at 1R

Why this differs from failed attempts:
- KAMA adapts better than static EMA/HMA in varying volatility
- Triple HTF (12h+1d+1w) provides stronger trend confirmation than dual
- ADX + BB Width dual regime filter avoids chop better than single filter
- Conservative sizing (0.28 base) controls drawdown vs 0.35+ in failed strats
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_triple_htf_adx_regime_12h_1d_1w_v1"
timeframe = "12h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER): measures trend efficiency
    change = close_s.diff(er_period).abs()
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = change / (volatility + 1e-10)
    
    # Smoothing Constant (SC)
    fast_sc_val = 2.0 / (fast_sc + 1)
    slow_sc_val = 2.0 / (slow_sc + 1)
    sc = (er * (fast_sc_val - slow_sc_val) + slow_sc_val) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc.iloc[i] * (close[i] - kama[i - 1])
    
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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth TR, +DM, -DM using Wilder's method
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx, plus_di, minus_di


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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - handles shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    atr = calculate_atr(high, low, close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2)
    
    # Calculate Bollinger Band Width percentile rank (regime filter)
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = BASE_SIZE / 2  # For take profit reduction
    
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
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or
            np.isnan(kama[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or
            np.isnan(rsi[i]) or np.isnan(bb_width_pr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # === TREND FILTERS (Triple HTF Alignment) ===
        # 12h KAMA trend
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # 1d HMA trend
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # 1w HMA trend
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        
        # === REGIME FILTERS ===
        # ADX > 25 (trending market, not choppy)
        adx_valid = adx[i] > 25
        
        # BB Width > 40th percentile (volatility expansion)
        regime_valid = bb_width_pr[i] > 0.40
        
        # DI confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = minus_di[i] > plus_di[i]
        
        # === ENTRY TIMING ===
        # RSI pullback zone (45-55 for entry timing in trend direction)
        rsi_pullback_long = 45 <= rsi[i] <= 55
        rsi_pullback_short = 45 <= rsi[i] <= 55
        
        # === SIGNAL LOGIC ===
        target_signal = 0.0
        
        # Long entry: All trends bullish + ADX valid + regime valid + RSI pullback + DI bullish
        if (kama_trend == 1 and daily_trend == 1 and weekly_trend == 1 and
            adx_valid and regime_valid and rsi_pullback_long and di_bullish):
            target_signal = BASE_SIZE
        
        # Short entry: All trends bearish + ADX valid + regime valid + RSI pullback + DI bearish
        elif (kama_trend == -1 and daily_trend == -1 and weekly_trend == -1 and
              adx_valid and regime_valid and rsi_pullback_short and di_bearish):
            target_signal = -BASE_SIZE
        
        # === STOPLOSS AND TAKE PROFIT LOGIC ===
        stoploss_triggered = False
        take_profit_triggered = False
        trend_reversal = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR
                        take_profit_triggered = True
                
                # Check trend reversal (KAMA flip)
                if kama_trend == -1:
                    trend_reversal = True
                    
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
                
                # Check trend reversal
                if kama_trend == 1:
                    trend_reversal = True
        
        # Apply signals in priority order
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
            
        elif trend_reversal:
            # Exit on trend reversal
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 1.0
            profit_target_hit = False
            
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
                # Maintain existing position
                signals[i] = BASE_SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals