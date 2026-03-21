#!/usr/bin/env python3
"""
EXPERIMENT #104 - MTF Supertrend+RSI+Chandelier+VolAdj Sizing (15m+1h+4h v2)
==================================================================================================
Hypothesis: Build on #040 (Sharpe=5.4) but add proper Chandelier exit and vol-adjusted sizing.
Key learnings from failures #101-#103:
- Chandelier alone failed (too much whipsaw) - need to combine with ATR trailing only AFTER profit
- Vol-adjusted sizing failed when applied naively - need smooth scaling, not binary switches
- Triple timeframe (15m+1h+4h) works better than dual (15m+4h)

Changes from #040:
- Add Chandelier exit (highest_high - 3*ATR(22)) but ONLY activate after 1R profit
- Volatility-adjusted position sizing: base_size * (target_vol / current_vol)
  - Low vol (ATR% < 1.5%): size = 0.35
  - Medium vol (1.5-3%): size = 0.25
  - High vol (>3%): size = 0.15
- Use mtf_data helper for PROPER 1h/4h alignment (no manual resampling!)
- Discrete signal levels: 0.0, ±0.15, ±0.25, ±0.35 to reduce churn costs
- Tighter RSI range: 35-55 for long, 45-65 for short (better pullback quality)
- ADX filter on 4h (not 1h) for stronger trend confirmation

Why this should beat current best (Sharpe=3.653):
- Vol-adjusted sizing reduces exposure during high-vol crashes (like 2022)
- Chandelier + ATR trailing combo gives better exit timing
- 4h ADX filter is more robust than 1h (less noise)
- Based on proven #040 foundation with risk management improvements
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_chandelier_voladj_15m_1h_4h_v2"
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
    
    wma1 = pd.Series(close).rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    ).values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
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
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
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
    for i in range(period - 1, n):
        if rolling_std[i] > 0:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
    
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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return np.nan_to_num(adx, nan=0.0)


def calculate_chandelier_exit(high, low, close, atr, period=22, multiplier=3.0):
    """Calculate Chandelier Exit (trailing stop based on highest high)"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    chandelier_long = np.zeros(n)
    chandelier_short = np.zeros(n)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        chandelier_long[i] = highest_high - multiplier * atr[i]
        chandelier_short[i] = lowest_low + multiplier * atr[i]
    
    return chandelier_long, chandelier_short


def get_volatility_regime(atr_pct):
    """
    Determine volatility regime based on ATR% (ATR / close * 100)
    Returns position size multiplier
    """
    if atr_pct < 1.5:
        return 0.35  # Low vol - full size
    elif atr_pct < 3.0:
        return 0.25  # Medium vol - reduced size
    else:
        return 0.15  # High vol - minimal size


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 15m indicators (entry timeframe) ==========
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # ATR% for volatility regime detection
    atr_pct_15m = np.zeros(n)
    for i in range(14, n):
        if close[i] > 0:
            atr_pct_15m[i] = (atr_15m[i] / close[i]) * 100
    
    # ========== Multi-timeframe using mtf_data helper ==========
    # 1h data for intermediate trend
    try:
        df_1h = get_htf_data(prices, '1h')
        if len(df_1h) > 0:
            hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
            st_1h_raw, st_dir_1h_raw = calculate_supertrend(
                df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, period=10, multiplier=3.0
            )
            adx_1h_raw = calculate_adx(
                df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, period=14
            )
            
            hma_1h = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
            st_dir_1h = align_htf_to_ltf(prices, df_1h, st_dir_1h_raw)
            adx_1h = align_htf_to_ltf(prices, df_1h, adx_1h_raw)
        else:
            hma_1h = np.zeros(n)
            st_dir_1h = np.zeros(n)
            adx_1h = np.zeros(n)
    except Exception:
        hma_1h = np.zeros(n)
        st_dir_1h = np.zeros(n)
        adx_1h = np.zeros(n)
    
    # 4h data for primary trend
    try:
        df_4h = get_htf_data(prices, '4h')
        if len(df_4h) > 0:
            hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
            st_dir_4h_raw = calculate_supertrend(
                df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=10, multiplier=3.0
            )[1]
            adx_4h_raw = calculate_adx(
                df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14
            )
            
            hma_4h = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
            st_dir_4h = align_htf_to_ltf(prices, df_4h, st_dir_4h_raw)
            adx_4h = align_htf_to_ltf(prices, df_4h, adx_4h_raw)
        else:
            hma_4h = np.zeros(n)
            st_dir_4h = np.zeros(n)
            adx_4h = np.zeros(n)
    except Exception:
        hma_4h = np.zeros(n)
        st_dir_4h = np.zeros(n)
        adx_4h = np.zeros(n)
    
    # ========== Chandelier Exit (15m) ==========
    chandelier_long, chandelier_short = calculate_chandelier_exit(
        high, low, close, atr_15m, period=22, multiplier=3.0
    )
    
    # ========== Generate signals ==========
    signals = np.zeros(n)
    
    # Position state tracking
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    profit_r = np.zeros(n)  # Profit in R multiples
    chandelier_active = np.zeros(n)  # Whether Chandelier exit is active
    
    # Parameters
    ATR_STOP_MULT = 2.0  # Initial stoploss
    TP_1R = 1.0  # Activate Chandelier at 1R profit
    TP_REDUCE = 2.0  # Reduce position at 2R profit
    ADX_4H_MIN = 20  # 4h ADX threshold
    ADX_1H_MIN = 18  # 1h ADX threshold
    RSI_LONG_MIN, RSI_LONG_MAX = 35, 55
    RSI_SHORT_MIN, RSI_SHORT_MAX = 45, 65
    ZSCORE_MAX = 2.0
    
    first_valid = max(100, 40)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0 or np.isnan(close[i]) or close[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        zscore_val = zscore_15m[i]
        atr_pct = atr_pct_15m[i]
        
        # ========== Check existing positions ==========
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_profit_r = profit_r[i - 1]
            prev_chandelier = chandelier_active[i - 1]
            
            # Calculate current profit in R
            if prev_side == 1:
                current_profit_r = (price - prev_entry) / (ATR_STOP_MULT * atr)
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                chandelier_stop = chandelier_long[i]
            else:
                current_profit_r = (prev_entry - price) / (ATR_STOP_MULT * atr)
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                chandelier_stop = chandelier_short[i]
            
            profit_r[i] = current_profit_r
            
            # Initial stoploss check (before 1R profit)
            if current_profit_r < TP_1R:
                if prev_side == 1 and price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    profit_r[i] = 0
                    chandelier_active[i] = 0
                    continue
                elif prev_side == -1 and price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    profit_r[i] = 0
                    chandelier_active[i] = 0
                    continue
            
            # Activate Chandelier exit after 1R profit
            if current_profit_r >= TP_1R and not prev_chandelier:
                chandelier_active[i] = 1
                prev_chandelier = 1
            
            # Chandelier exit check (after activation)
            if prev_chandelier:
                if prev_side == 1 and price < chandelier_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    profit_r[i] = 0
                    chandelier_active[i] = 0
                    continue
                elif prev_side == -1 and price > chandelier_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    profit_r[i] = 0
                    chandelier_active[i] = 0
                    continue
            
            # Reduce position at 2R profit
            if current_profit_r >= TP_REDUCE:
                if prev_side == 1:
                    signals[i] = 0.175  # Half of 0.35
                else:
                    signals[i] = -0.175
                position_side[i] = prev_side
                entry_price[i] = prev_entry
                profit_r[i] = current_profit_r
                chandelier_active[i] = prev_chandelier
                continue
            
            # Hold position
            signals[i] = signals[i - 1]
            position_side[i] = prev_side
            entry_price[i] = prev_entry
            profit_r[i] = current_profit_r
            chandelier_active[i] = prev_chandelier
            continue
        
        # ========== Entry logic ==========
        # 4h trend filter (primary)
        trend_4h = 0
        if hma_4h[i] > 0 and close[i] > hma_4h[i]:
            trend_4h = 1
        elif hma_4h[i] > 0 and close[i] < hma_4h[i]:
            trend_4h = -1
        
        # 1h trend filter (intermediate)
        trend_1h = 0
        if hma_1h[i] > 0 and close[i] > hma_1h[i]:
            trend_1h = 1
        elif hma_1h[i] > 0 and close[i] < hma_1h[i]:
            trend_1h = -1
        
        # ADX filters
        adx_4h_ok = adx_4h[i] >= ADX_4H_MIN
        adx_1h_ok = adx_1h[i] >= ADX_1H_MIN
        
        # Get position size based on volatility regime
        base_size = get_volatility_regime(atr_pct)
        
        # Long entry
        if (trend_4h == 1 and trend_1h == 1 and 
            st_direction_15m[i] == 1 and st_dir_1h[i] == 1 and
            adx_4h_ok and adx_1h_ok and
            RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and
            abs(zscore_val) < ZSCORE_MAX):
            
            signals[i] = base_size
            position_side[i] = 1
            entry_price[i] = price
            profit_r[i] = 0
            chandelier_active[i] = 0
        
        # Short entry
        elif (trend_4h == -1 and trend_1h == -1 and 
              st_direction_15m[i] == -1 and st_dir_1h[i] == -1 and
              adx_4h_ok and adx_1h_ok and
              RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and
              abs(zscore_val) < ZSCORE_MAX):
            
            signals[i] = -base_size
            position_side[i] = -1
            entry_price[i] = price
            profit_r[i] = 0
            chandelier_active[i] = 0
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals