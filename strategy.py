#!/usr/bin/env python3
"""
EXPERIMENT #012 - DEMA Trend + Stochastic Entry + ADX/Z-score Filter
=======================================================================
Hypothesis: DEMA (Double EMA) responds faster to trend changes than HMA/KAMA,
reducing lag in trend detection. Combined with Stochastic for precise pullback
entries and ADX strength filter, this should improve entry timing while avoiding
weak trend periods. Z-score filter prevents entering at extreme extensions.

Key differences from previous strategies:
- DEMA(8/21) crossover instead of HMA/KAMA for faster trend response
- Stochastic(14,3,3) instead of RSI for entry timing (different signal type)
- ADX(14) strength filter - only trade when ADX > 20 (strong trend)
- Z-score(20) filter - avoid entries when price > 2 std from mean
- Multi-timeframe: 4h DEMA trend + 1h Stochastic entries
- ATR trailing stop with 2.0*ATR distance

Why this might beat Sharpe=2.931:
- DEMA has less lag than HMA for trend changes
- Stochastic more sensitive to pullbacks than RSI
- ADX filter avoids choppy/weak trend periods (major source of losses)
- Z-score prevents buying tops/selling bottoms
"""

import numpy as np
import pandas as pd

name = "mtf_dema_stoch_adx_zscore_v1"
timeframe = "1h"
leverage = 1.0


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    ema1 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False).mean().values
    dema = 2 * ema1 - ema2
    return dema


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    n = len(close)
    k = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(k_period - 1, n):
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        highest_high = np.max(high[i - k_period + 1:i + 1])
        
        if highest_high > lowest_low:
            k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
        else:
            k[i] = 50.0
    
    # Calculate %D (signal line)
    for i in range(d_period - 1, n):
        d[i] = np.mean(k[i - d_period + 1:i + 1])
    
    return k, d


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        # True Range
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        # Directional Movement
        plus_dm[i] = max(0, high[i] - high[i - 1]) if (high[i] - high[i - 1]) > (low[i - 1] - low[i]) else 0
        minus_dm[i] = max(0, low[i - 1] - low[i]) if (low[i - 1] - low[i]) > (high[i] - high[i - 1]) else 0
    
    # Smooth with Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Initialize first values
    sum_tr = np.sum(tr[1:period + 1])
    sum_plus_dm = np.sum(plus_dm[1:period + 1])
    sum_minus_dm = np.sum(minus_dm[1:period + 1])
    
    if sum_tr > 0:
        plus_di[period] = 100 * sum_plus_dm / sum_tr
        minus_di[period] = 100 * sum_minus_dm / sum_tr
    
    if plus_di[period] + minus_di[period] > 0:
        dx[period] = 100 * abs(plus_di[period] - minus_di[period]) / (plus_di[period] + minus_di[period])
    
    adx[period] = dx[period]
    
    for i in range(period + 1, n):
        # Wilder's smoothing
        sum_tr = sum_tr - tr[i - period] + tr[i]
        sum_plus_dm = sum_plus_dm - plus_dm[i - period] + plus_dm[i]
        sum_minus_dm = sum_minus_dm - minus_dm[i - period] + minus_dm[i]
        
        if sum_tr > 0:
            plus_di[i] = 100 * sum_plus_dm / sum_tr
            minus_di[i] = 100 * sum_minus_dm / sum_tr
        
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
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


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    n = len(close)
    zscore = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            zscore[i] = (close[i] - mean) / std
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    stoch_k_1h, stoch_d_1h = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    
    # 4h indicators for trend (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # Calculate 4h DEMA for trend
    dema_fast_4h = calculate_dema(c_4h, period=8)
    dema_slow_4h = calculate_dema(c_4h, period=21)
    
    # Calculate 4h ADX for trend strength
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # 4h trend direction based on DEMA crossover
    trend_4h = np.zeros(len(c_4h))
    for i in range(21, len(c_4h)):
        if not np.isnan(dema_fast_4h[i]) and not np.isnan(dema_slow_4h[i]):
            if dema_fast_4h[i] > dema_slow_4h[i]:
                trend_4h[i] = 1  # Bullish
            elif dema_fast_4h[i] < dema_slow_4h[i]:
                trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    adx_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
            adx_1h[i] = adx_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # Stochastic thresholds for pullback entries
    STCH_LONG_ENTRY = 35   # Enter long on oversold pullback in uptrend
    STCH_SHORT_ENTRY = 65  # Enter short on overbought rally in downtrend
    STCH_EXIT = 50         # Exit when stochastic crosses middle
    
    # ADX threshold for trend strength
    ADX_MIN = 20           # Only trade when ADX > 20 (strong trend)
    
    # Z-score thresholds for mean reversion filter
    ZSCORE_MAX = 1.8       # Don't enter when price > 1.8 std from mean
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(80, 21, 14, 20)  # Wait for all indicators
    
    # Track entry prices for trailing stop logic
    entry_price = np.zeros(n)
    position_type = np.zeros(n)  # 1=long, -1=short, 0=none
    
    for i in range(first_valid, n):
        if np.isnan(stoch_k_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        adx_val = adx_1h[i]
        stoch_k = stoch_k_1h[i]
        stoch_d = stoch_d_1h[i]
        zscore = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # ADX filter - only trade when trend is strong
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            continue
        
        # Z-score filter - avoid extreme extensions
        if abs(zscore) > ZSCORE_MAX:
            signals[i] = 0.0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            continue
        
        # Check trailing stop for existing positions first
        if i > 0 and position_type[i - 1] != 0 and entry_price[i - 1] > 0:
            prev_entry = entry_price[i - 1]
            prev_pos = position_type[i - 1]
            stoploss_distance = ATR_STOP_MULT * atr
            
            if prev_pos == 1:  # Long position
                stoploss_price = prev_entry - stoploss_distance
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_type[i] = 0
                    entry_price[i] = 0
                    continue
            elif prev_pos == -1:  # Short position
                stoploss_price = prev_entry + stoploss_distance
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_type[i] = 0
                    entry_price[i] = 0
                    continue
        
        if trend == 1:  # 4h uptrend
            # Check for long entry
            if stoch_k < STCH_LONG_ENTRY and stoch_k > stoch_d:
                # Stochastic crossing up from oversold - full position
                signals[i] = SIZE_FULL
                entry_price[i] = price
                position_type[i] = 1
            elif stoch_k < 45 and signals[i - 1] > 0:
                # Hold existing long with reduced size if stochastic rising
                if stoch_k > stoch_d:
                    signals[i] = SIZE_HALF
                    position_type[i] = 1
                else:
                    signals[i] = signals[i - 1]
                    position_type[i] = position_type[i - 1]
                    entry_price[i] = entry_price[i - 1]
            elif stoch_k > STCH_EXIT and signals[i - 1] > 0:
                # Exit long when stochastic crosses above middle
                signals[i] = 0.0
                position_type[i] = 0
                entry_price[i] = 0
            else:
                # Hold or exit
                if signals[i - 1] > 0:
                    signals[i] = signals[i - 1]
                    position_type[i] = position_type[i - 1]
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_type[i] = 0
                    entry_price[i] = 0
        elif trend == -1:  # 4h downtrend
            # Check for short entry
            if stoch_k > STCH_SHORT_ENTRY and stoch_k < stoch_d:
                # Stochastic crossing down from overbought - full short
                signals[i] = -SIZE_FULL
                entry_price[i] = price
                position_type[i] = -1
            elif stoch_k > 55 and signals[i - 1] < 0:
                # Hold existing short with reduced size if stochastic falling
                if stoch_k < stoch_d:
                    signals[i] = -SIZE_HALF
                    position_type[i] = -1
                else:
                    signals[i] = signals[i - 1]
                    position_type[i] = position_type[i - 1]
                    entry_price[i] = entry_price[i - 1]
            elif stoch_k < STCH_EXIT and signals[i - 1] < 0:
                # Exit short when stochastic crosses below middle
                signals[i] = 0.0
                position_type[i] = 0
                entry_price[i] = 0
            else:
                # Hold or exit
                if signals[i - 1] < 0:
                    signals[i] = signals[i - 1]
                    position_type[i] = position_type[i - 1]
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_type[i] = 0
                    entry_price[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_type[i] = 0
            entry_price[i] = 0
    
    return signals