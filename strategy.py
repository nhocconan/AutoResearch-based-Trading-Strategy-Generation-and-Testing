#!/usr/bin/env python3
"""
EXPERIMENT #010 - Supertrend + RSI Pullback + ATR Volatility Regime Filter
===========================================================================
Hypothesis: Supertrend (4h) provides cleaner trend signals than Donchian with less whipsaw.
RSI pullback entries (1h) work well in trending markets. ATR volatility regime filter
avoids trading during extreme volatility periods (which caused -80% DD in exp#008).

Key innovations vs mtf_kama_macd_adx_atr_v1 (Sharpe=2.139):
- Supertrend(4h) instead of KAMA for binary trend direction (less lag)
- RSI(14) pullback entries instead of MACD (faster entry timing)
- ATR volatility percentile filter to avoid extreme volatility regimes
- Conservative position sizing: max 0.30 (vs 0.35-0.40 in failed strategies)
- Discrete signal levels (0.0, ±0.20, ±0.30) to minimize churn costs

Why this might beat Sharpe=2.139:
- Supertrend proven in exp#002 (Sharpe=1.278) but failed due to position sizing
- RSI pullback logic from mtf_hma_rsi_zscore_v1 (Sharpe=5.4 baseline)
- ATR regime filter prevents trading during crash volatility (fixes exp#008 -80% DD)
- Multi-timeframe logic (4h trend + 1h entry) proven to 2x Sharpe
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_rsi_atr_regime_v1"
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
    """Calculate Supertrend indicator for binary trend direction"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 for bullish, -1 for bearish
    
    for i in range(period, n):
        if atr[i] == 0:
            supertrend[i] = close[i]
            continue
            
        upper_band = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band = (high[i] + low[i]) / 2 - multiplier * atr[i]
        
        if trend[i - 1] == 1:
            if close[i] < lower_band:
                trend[i] = -1
                supertrend[i] = upper_band
            else:
                trend[i] = 1
                supertrend[i] = max(lower_band, supertrend[i - 1] if i > period else lower_band)
        else:
            if close[i] > upper_band:
                trend[i] = 1
                supertrend[i] = lower_band
            else:
                trend[i] = -1
                supertrend[i] = min(upper_band, supertrend[i - 1] if i > period else upper_band)
    
    return supertrend, trend


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


def calculate_atr_percentile(atr, lookback=50):
    """Calculate ATR percentile for volatility regime detection"""
    n = len(atr)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = atr[i - lookback + 1:i + 1]
        rank = np.sum(window < atr[i])
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    atr_pct_1h = calculate_atr_percentile(atr_1h, lookback=50)
    
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
    _, trend_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.30   # Full position in good conditions (conservative vs 0.35)
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    RSI_EXIT = 50         # Exit when RSI crosses midline against position
    
    # ATR volatility regime filter
    ATR_PCT_HIGH = 0.85   # Don't trade if ATR in top 15% (extreme volatility)
    ATR_PCT_LOW = 0.15    # Reduce size if ATR in bottom 15% (low volatility = chop)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 14, 50)  # Wait for all indicators
    
    # Track entry prices for trailing stop logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(atr_pct_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        atr_pct = atr_pct_1h[i]
        price = close[i]
        
        # ATR volatility regime filter - avoid extreme volatility
        if atr_pct > ATR_PCT_HIGH:
            # Extreme volatility - close all positions
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            
            if prev_side == 1:  # Long position
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                # Also check RSI exit signal
                rsi_exit = rsi_val < RSI_EXIT
                
                if price < stoploss_price or rsi_exit:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
            elif prev_side == -1:  # Short position
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                # Also check RSI exit signal
                rsi_exit = rsi_val > RSI_EXIT
                
                if price > stoploss_price or rsi_exit:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
        
        # Low volatility regime - reduce position size
        size_multiplier = 1.0
        if atr_pct < ATR_PCT_LOW:
            size_multiplier = 0.67  # Reduce to 2/3 size in low vol
        
        if trend == 1:  # 4h uptrend
            if rsi_val < RSI_LONG_ENTRY:
                # Pullback entry - full position (adjusted for vol regime)
                signals[i] = SIZE_FULL * size_multiplier
                position_side[i] = 1
                entry_price[i] = price
            elif rsi_val < 50:
                # Moderate pullback - half position
                signals[i] = SIZE_HALF * size_multiplier
                position_side[i] = 1
                entry_price[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            if rsi_val > RSI_SHORT_ENTRY:
                # Rally entry - full short
                signals[i] = -SIZE_FULL * size_multiplier
                position_side[i] = -1
                entry_price[i] = price
            elif rsi_val > 50:
                # Moderate rally - half short
                signals[i] = -SIZE_HALF * size_multiplier
                position_side[i] = -1
                entry_price[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
        else:  # No clear trend (flat)
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
    
    return signals