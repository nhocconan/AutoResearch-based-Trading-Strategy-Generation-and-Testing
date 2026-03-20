#!/usr/bin/env python3
"""
EXPERIMENT #016 - Supertrend + Z-Score Pullback + ADX Filter + ATR Stop
========================================================================
Hypothesis: Supertrend provides cleaner trend signals than MA/KAMA. Combined with
Z-score mean reversion for entry timing (buy pullbacks in uptrend, sell rallies in
downtrend) and ADX strength filter to avoid choppy markets. ATR trailing stop manages
risk dynamically.

Key innovations vs current best (mtf_keltner_rsi_adx_v1, Sharpe=4.452):
- Supertrend instead of Keltner for trend (less whipsaw, clearer signals)
- Z-score on price deviation from rolling mean (better entry timing than RSI)
- ADX > 25 filter ensures we only trade when trend has momentum
- Multi-timeframe: 4h Supertrend trend + 1h Z-score entries
- Discrete signal levels (0, ±0.25, ±0.35) to reduce churn costs

Why this might beat Sharpe=4.452:
- Supertrend adapts to volatility better than Keltner channels
- Z-score captures mean reversion within trends more precisely
- ADX filter avoids false signals in consolidation
- Proper ATR stoploss prevents large drawdowns
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_zscore_adx_v1"
timeframe = "1h"
leverage = 1.0


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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    for i in range(period, n):
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1
        else:
            if trend[i - 1] == 1:
                if close[i] < lower_band[i]:
                    supertrend[i] = upper_band[i]
                    trend[i] = -1
                else:
                    supertrend[i] = max(lower_band[i], supertrend[i - 1])
                    trend[i] = 1
            else:
                if close[i] > upper_band[i]:
                    supertrend[i] = lower_band[i]
                    trend[i] = 1
                else:
                    supertrend[i] = min(upper_band[i], supertrend[i - 1])
                    trend[i] = -1
    
    return supertrend, trend


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # Smooth TR, +DM, -DM using Wilder's method
    tr_smooth = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    tr_smooth[period - 1] = np.sum(tr[1:period])
    plus_dm_smooth[period - 1] = np.sum(plus_dm[1:period])
    minus_dm_smooth[period - 1] = np.sum(minus_dm[1:period])
    
    for i in range(period, n):
        tr_smooth[i] = tr_smooth[i - 1] - tr_smooth[i - 1] / period + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - plus_dm_smooth[i - 1] / period + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - minus_dm_smooth[i - 1] / period + minus_dm[i]
    
    for i in range(period, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
        
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Smooth DX to get ADX
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean"""
    n = len(close)
    zscore = np.zeros(n)
    
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    mask = std > 0
    zscore[mask] = (close[mask] - mean[mask]) / std[mask]
    
    return zscore


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    zscore_1h = calculate_zscore(close, period=20)
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    
    # 4h Supertrend for trend filter (resample 1h → 4h)
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
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Calculate 4h Supertrend
    supertrend_4h, trend_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Calculate 4h ADX for trend strength
    adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    adx_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
        if idx_4h < len(adx_4h):
            adx_1h[i] = adx_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.25   # Reduced position in marginal conditions
    
    # Z-score thresholds for pullback entries
    ZSCORE_LONG_ENTRY = -0.8   # Enter long on pullback (price below mean)
    ZSCORE_SHORT_ENTRY = 0.8   # Enter short on rally (price above mean)
    ZSCORE_EXIT = 0.3          # Exit when mean reversion complete
    
    # ADX threshold for trend strength
    ADX_MIN = 25               # Only trade when ADX > 25 (strong trend)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 40, 28, 20)  # Wait for all indicators
    
    # Track entry prices for trailing stop logic
    entry_price_long = np.zeros(n)
    entry_price_short = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(zscore_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(trend_1h[i]) or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        zscore_val = zscore_1h[i]
        adx_val = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        rsi_val = rsi_1h[i]
        
        # ADX filter - only trade when trend has momentum
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            continue
        
        if trend == 1:  # 4h uptrend
            # Check trailing stop for existing long first
            if i > 0 and signals[i - 1] > 0:
                if entry_price_long[i - 1] > 0:
                    entry_price = entry_price_long[i - 1]
                    highest_since_entry[i] = max(highest_since_entry[i - 1], price)
                    stoploss_price = entry_price - ATR_STOP_MULT * atr
                    
                    # Trail stop: move stop up as price rises
                    trailing_stop = highest_since_entry[i] - ATR_STOP_MULT * atr
                    stoploss_price = max(stoploss_price, trailing_stop)
                    
                    if price < stoploss_price:
                        signals[i] = 0.0  # Stoploss triggered
                        entry_price_long[i] = 0
                        highest_since_entry[i] = 0
                    elif zscore_val > ZSCORE_EXIT:
                        # Take profit: reduce to half position
                        signals[i] = SIZE_HALF
                        entry_price_long[i] = entry_price
                        highest_since_entry[i] = highest_since_entry[i]
                    else:
                        signals[i] = signals[i - 1]  # Hold position
                        entry_price_long[i] = entry_price
                        highest_since_entry[i] = highest_since_entry[i]
                else:
                    signals[i] = 0.0
            else:
                # Look for entry on pullback
                if zscore_val < ZSCORE_LONG_ENTRY:
                    # Strong pullback - full position
                    signals[i] = SIZE_FULL
                    entry_price_long[i] = price
                    highest_since_entry[i] = price
                elif zscore_val < 0 and rsi_val < 50:
                    # Moderate pullback - half position
                    signals[i] = SIZE_HALF
                    entry_price_long[i] = price
                    highest_since_entry[i] = price
                else:
                    signals[i] = 0.0
                    
        elif trend == -1:  # 4h downtrend
            # Check trailing stop for existing short first
            if i > 0 and signals[i - 1] < 0:
                if entry_price_short[i - 1] > 0:
                    entry_price = entry_price_short[i - 1]
                    lowest_since_entry[i] = min(lowest_since_entry[i - 1], price) if lowest_since_entry[i - 1] > 0 else price
                    stoploss_price = entry_price + ATR_STOP_MULT * atr
                    
                    # Trail stop: move stop down as price falls
                    trailing_stop = lowest_since_entry[i] + ATR_STOP_MULT * atr
                    stoploss_price = min(stoploss_price, trailing_stop)
                    
                    if price > stoploss_price:
                        signals[i] = 0.0  # Stoploss triggered
                        entry_price_short[i] = 0
                        lowest_since_entry[i] = 0
                    elif zscore_val < -ZSCORE_EXIT:
                        # Take profit: reduce to half position
                        signals[i] = -SIZE_HALF
                        entry_price_short[i] = entry_price
                        lowest_since_entry[i] = lowest_since_entry[i]
                    else:
                        signals[i] = signals[i - 1]  # Hold position
                        entry_price_short[i] = entry_price
                        lowest_since_entry[i] = lowest_since_entry[i]
                else:
                    signals[i] = 0.0
            else:
                # Look for entry on rally
                if zscore_val > ZSCORE_SHORT_ENTRY:
                    # Strong rally - full short
                    signals[i] = -SIZE_FULL
                    entry_price_short[i] = price
                    lowest_since_entry[i] = price
                elif zscore_val > 0 and rsi_val > 50:
                    # Moderate rally - half short
                    signals[i] = -SIZE_HALF
                    entry_price_short[i] = price
                    lowest_since_entry[i] = price
                else:
                    signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
    
    return signals