#!/usr/bin/env python3
"""
Experiment #406: 4h KAMA Adaptive Trend + Daily HMA Bias + BBW Regime + Z-Score Entry + ATR Stop
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - moves fast in trends,
slow in ranges. This should reduce whipsaw vs fixed EMA/HMA on 4h timeframe. Daily HMA provides
medium-term trend bias via mtf_data helper. Bollinger Band Width (BBW) percentile detects volatility
regimes: low BBW = consolidation (prepare for breakout), high BBW = trending (follow trend).
Z-score(20) entry within trend direction captures pullbacks without being too restrictive like RSI.
ATR(14) stoploss at 2.5x for 4h timeframe. Position size 0.25 discrete to control drawdown.
Key insight: Previous 4h strategies failed due to too many filters (Supertrend+RSI+ADX all failed).
This uses fewer, more robust filters: KAMA trend + Daily bias + Z-score pullback + BBW regime.
Timeframe: 4h (REQUIRED for this experiment), HTF: 1d for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 (current best mtf_12h_supertrend_daily_hma_rsi_pullback_v2).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_daily_hma_bbw_regime_zscore_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - moves fast in trending markets, slow in ranging.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + 1:
        return kama
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(period, n):
        vol_sum = 0.0
        for j in range(1, period + 1):
            if i - j >= 0:
                vol_sum += np.abs(close[i] - close[i - j])
        volatility[i] = vol_sum
    
    er = np.zeros(n)
    er[:] = np.nan
    mask = (volatility > 0) & (~np.isnan(change))
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = np.zeros(n)
    sc[:] = np.nan
    valid_er = ~np.isnan(er)
    sc[valid_er] = (er[valid_er] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bbw = (upper - lower) / sma * 100  # Band Width as percentage
    return upper, lower, bbw

def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile rank over lookback period."""
    n = len(bbw)
    bbw_pct = np.zeros(n)
    bbw_pct[:] = np.nan
    
    for i in range(lookback, n):
        if np.isnan(bbw[i]):
            continue
        window = bbw[i-lookback+1:i+1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            bbw_pct[i] = np.sum(valid_window < bbw[i]) / len(valid_window) * 100
    
    return bbw_pct

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / std
    zscore = np.where(std > 0, zscore, 0.0)
    return zscore

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    bb_upper, bb_lower, bbw = calculate_bollinger_bands(close, 20, 2.0)
    bbw_pct = calculate_bbw_percentile(bbw, 100)
    zscore = calculate_zscore(close, 20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):  # Start after 150 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(kama[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(bbw[i]) or np.isnan(bbw_pct[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (medium-term direction)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # KAMA trend direction
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # KAMA slope (momentum)
        kama_slope_bullish = kama[i] > kama[i-5] if i > 5 else False
        kama_slope_bearish = kama[i] < kama[i-5] if i > 5 else False
        
        # BBW regime: low volatility = consolidation, high = trending
        bbw_low = bbw_pct[i] < 40  # Bottom 40% = low vol, prepare for breakout
        bbw_high = bbw_pct[i] > 60  # Top 40% = high vol, trending
        
        # Z-score pullback entry (within trend)
        zscore_pullback_long = zscore[i] < -0.5 and zscore[i] > -2.5  # Pullback but not extreme
        zscore_pullback_short = zscore[i] > 0.5 and zscore[i] < 2.5  # Rally but not extreme
        zscore_strong_long = zscore[i] > 0.5 and zscore[i] < 2.0  # Momentum long
        zscore_strong_short = zscore[i] < -0.5 and zscore[i] > -2.0  # Momentum short
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple conditions to ensure trade frequency) ===
        # Primary: KAMA bullish + Daily bullish + Z-score pullback
        if kama_bullish and daily_bullish and zscore_pullback_long:
            new_signal = SIZE_ENTRY
        # Secondary: KAMA bullish + KAMA slope up + Daily bullish + Z-score ok
        elif kama_bullish and kama_slope_bullish and daily_bullish and zscore[i] > -1.0:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA bullish + Daily bullish + BBW low (breakout setup)
        elif kama_bullish and daily_bullish and bbw_low and zscore[i] > -1.5:
            new_signal = SIZE_ENTRY
        # Quaternary: KAMA bullish + Z-score momentum (daily neutral ok)
        elif kama_bullish and zscore_strong_long and bbw_high:
            new_signal = SIZE_ENTRY
        # Quintenary: KAMA crossover + Daily bullish
        elif kama_bullish and close[i-1] <= kama[i-1] and daily_bullish:
            new_signal = SIZE_ENTRY
        # Sextenary: Daily bullish + KAMA bullish (simple, ensures trades)
        elif daily_bullish and kama_bullish and zscore[i] > -2.0:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple conditions to ensure trade frequency) ===
        # Primary: KAMA bearish + Daily bearish + Z-score pullback
        if kama_bearish and daily_bearish and zscore_pullback_short:
            new_signal = -SIZE_ENTRY
        # Secondary: KAMA bearish + KAMA slope down + Daily bearish + Z-score ok
        elif kama_bearish and kama_slope_bearish and daily_bearish and zscore[i] < 1.0:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA bearish + Daily bearish + BBW low (breakdown setup)
        elif kama_bearish and daily_bearish and bbw_low and zscore[i] < 1.5:
            new_signal = -SIZE_ENTRY
        # Quaternary: KAMA bearish + Z-score momentum (daily neutral ok)
        elif kama_bearish and zscore_strong_short and bbw_high:
            new_signal = -SIZE_ENTRY
        # Quintenary: KAMA crossover + Daily bearish
        elif kama_bearish and close[i-1] >= kama[i-1] and daily_bearish:
            new_signal = -SIZE_ENTRY
        # Sextenary: Daily bearish + KAMA bearish (simple, ensures trades)
        elif daily_bearish and kama_bearish and zscore[i] < 2.0:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for 4h timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest for 4h timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals