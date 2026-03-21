#!/usr/bin/env python3
"""
EXPERIMENT #036 - MTF HMA+Supertrend+RSI+ATR Dynamic Sizing (15m+4h Clean v2)
==================================================================================================
Hypothesis: Experiments #031, #034, #035 proved 15m entries with 4h trend filter works best.
Current #040 does manual resampling which caused crashes in #027-#030, #034.

Key changes from #040:
- USE mtf_data helper (CRITICAL - 46 strategies failed audit without this)
- Timeframe: 15m entries + 4h trend (proven Sharpe > 7.5 in winning strategies)
- Remove KAMA/ADX/BBW filters (overfitting, caused failures in #032, #035)
- ATR-based dynamic position sizing: size = base_size * (target_vol / current_vol)
- Simpler state tracking (fewer bugs than #040's complex TP/trail logic)
- Position size: 0.30 base (slightly lower than 0.35 for safety)
- Stoploss: 2.5*ATR (wider than #040's 2.0*ATR to avoid premature exits)
- RSI thresholds: 35-65 (wider range for more trade opportunities)

Why this should beat #040:
- Uses mtf_data helper (avoids indexing bugs that crashed #027-#030, #034)
- Simpler logic = fewer bugs
- Dynamic sizing reduces risk during high volatility periods
- Based on proven winning combinations from #031, #034, #035
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_supertrend_rsi_atr_dynamic_15m_4h_v2"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).rolling(window=half_period).apply(
        lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period).apply(
        lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    ).values
    
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).rolling(window=sqrt_period).apply(
        lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    ).values
    
    return np.nan_to_num(hma, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return np.nan_to_num(rsi, nan=50.0)


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    for i in range(n):
        if rolling_std[i] > 0:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
        else:
            zscore[i] = 0
    
    return np.nan_to_num(zscore, nan=0.0)


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # 4h trend filters using mtf_data helper (CRITICAL - avoids indexing bugs)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h indicators
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    _, st_direction_4h = calculate_supertrend(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=10,
        multiplier=3.0
    )
    
    # Align 4h indicators to 15m timeframe (auto shift for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    st_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    BASE_SIZE = 0.30
    TARGET_VOL = 0.02  # Target 2% daily volatility
    
    # RSI thresholds for pullback entries (wider range for more opportunities)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 65
    
    # Z-score threshold for mean reversion filter
    ZSCORE_MAX = 2.5
    
    # ATR stoploss multiplier (wider to avoid premature exits)
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 50)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    stoploss_price = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filters
        trend_4h = 0
        if hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i]:
            trend_4h = 1
        elif hma_4h_aligned[i] > 0 and close[i] < hma_4h_aligned[i]:
            trend_4h = -1
        
        st_trend_4h = st_4h_aligned[i]
        
        # Dynamic position sizing based on volatility
        current_vol = atr_15m[i] / close[i] if close[i] > 0 else 0.02
        vol_adjustment = min(2.0, max(0.5, TARGET_VOL / current_vol)) if current_vol > 0 else 1.0
        dynamic_size = BASE_SIZE * vol_adjustment
        dynamic_size = min(0.40, max(0.15, dynamic_size))  # Clamp between 0.15 and 0.40
        
        # Check stoploss for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1]
            prev_stop = stoploss_price[i - 1]
            
            # Stoploss check
            if prev_side == 1 and close[i] < prev_stop:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                continue
            elif prev_side == -1 and close[i] > prev_stop:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                stoploss_price[i] = 0
                continue
            
            # Hold position - check if trend still valid
            if prev_side == 1:
                if trend_4h == 1 and st_trend_4h == 1:
                    signals[i] = dynamic_size
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    stoploss_price[i] = prev_stop
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    stoploss_price[i] = 0
            elif prev_side == -1:
                if trend_4h == -1 and st_trend_4h == -1:
                    signals[i] = -dynamic_size
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    stoploss_price[i] = prev_stop
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    stoploss_price[i] = 0
            continue
        
        # Entry logic: 4h trend + 15m RSI pullback + Z-score filter
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        
        # Bullish entry: 4h uptrend + 15m RSI pullback
        if trend_4h == 1 and st_trend_4h == 1:
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                abs(zscore_val) < ZSCORE_MAX):
                signals[i] = dynamic_size
                position_side[i] = 1
                entry_price[i] = close[i]
                stoploss_price[i] = close[i] - ATR_STOP_MULT * atr_15m[i]
        
        # Bearish entry: 4h downtrend + 15m RSI pullback
        elif trend_4h == -1 and st_trend_4h == -1:
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                abs(zscore_val) < ZSCORE_MAX):
                signals[i] = -dynamic_size
                position_side[i] = -1
                entry_price[i] = close[i]
                stoploss_price[i] = close[i] + ATR_STOP_MULT * atr_15m[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals