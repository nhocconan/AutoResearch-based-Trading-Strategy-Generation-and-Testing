#!/usr/bin/env python3
"""
EXPERIMENT #004 - Keltner Channel Trend + DEMA Crossover + Stochastic Entry + Volume Filter
============================================================================================
Hypothesis: Keltner Channels (volatility-based) provide cleaner trend signals than Donchian
(price-based). DEMA crossover gives faster response than HMA. Stochastic entry timing
catches pullbacks better than RSI. Volume filter confirms breakout validity.

Key differences from mtf_donchian_hma_rsi_zscore_v1:
- Keltner (EMA+ATR) instead of Donchian - smoother volatility-adaptive channels
- DEMA(8/21) instead of HMA(21) - faster trend response with less lag
- Stochastic(14,3,3) instead of RSI(14) - better for identifying pullback extremes
- Volume filter (20-period MA) to confirm breakout validity
- Discrete position sizing (0.0, ±0.25, ±0.35) to minimize churn costs
- ATR trailing stop at 2.5*ATR with proper entry price tracking

Why this might beat Sharpe=1.278:
- Keltner adapts to volatility regimes better than fixed Donchian periods
- DEMA responds faster to trend changes than HMA
- Stochastic oversold/overbought levels clearer for pullback entries
- Volume confirmation reduces false breakout entries
- Multi-signal confluence reduces whipsaw trades
"""

import numpy as np
import pandas as pd

name = "mtf_keltner_dema_stoch_volume_v1"
timeframe = "1h"
leverage = 1.0


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False).mean().values
    
    dema = 2 * ema1 - ema2
    dema[:period] = np.nan
    
    return dema


def calculate_keltner(high, low, close, ema_period=20, atr_period=10, multiplier=2.0):
    """Calculate Keltner Channel - volatility-adaptive trend channel"""
    n = len(close)
    
    # EMA for center line
    ema = pd.Series(close).ewm(span=ema_period, adjust=False).mean().values
    
    # ATR calculation
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False).mean().values
    
    # Keltner bands
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    
    return upper, lower, ema, atr


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator for entry timing"""
    n = len(close)
    
    lowest_low = pd.Series(low).rolling(window=k_period, min_periods=k_period).min().values
    highest_high = pd.Series(high).rolling(window=k_period, min_periods=k_period).max().values
    
    stoch_k = np.zeros(n)
    mask = highest_high > lowest_low
    stoch_k[mask] = 100 * (close[mask] - lowest_low[mask]) / (highest_high[mask] - lowest_low[mask])
    
    stoch_d = pd.Series(stoch_k).rolling(window=d_period, min_periods=d_period).mean().values
    
    return stoch_k, stoch_d


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume filter"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


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
    
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices.get("volume", np.ones(len(close))).values
    n = len(close)
    
    # Primary indicators
    dema_fast = calculate_dema(close, period=8)
    dema_slow = calculate_dema(close, period=21)
    keltner_upper, keltner_lower, keltner_mid, atr = calculate_keltner(high, low, close, ema_period=20, atr_period=10, multiplier=2.0)
    stoch_k, stoch_d = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    volume_sma = calculate_volume_sma(volume, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in strong conditions
    SIZE_HALF = 0.25   # Reduced position in marginal conditions
    
    # Entry thresholds
    STOCH_OVERSOLD = 25   # Enter long when stochastic oversold
    STOCH_OVERBOUGHT = 75  # Enter short when stochastic overbought
    VOLUME_RATIO = 1.2     # Volume must be 20% above average for confirmation
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # Minimum periods for all indicators
    first_valid = max(21, 20, 14, 20)  # DEMA slow, Keltner, Stochastic, Volume
    
    signals = np.zeros(n)
    
    # Track position state for trailing stop
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    
    for i in range(first_valid, n):
        # Check for NaN values
        if np.isnan(dema_fast[i]) or np.isnan(dema_slow[i]) or np.isnan(keltner_upper[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_val = atr_14[i] if not np.isnan(atr_14[i]) else 0
        vol_ratio = volume[i] / volume_sma[i] if volume_sma[i] > 0 else 1.0
        
        # ATR filter - avoid trading when extremely volatile
        if atr_val > 0 and atr_val / price > 0.05:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            
            if prev_side == 1:  # Long position
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_val
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
            elif prev_side == -1:  # Short position
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_val
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
        
        # Determine trend from Keltner position and DEMA crossover
        keltner_bullish = price > keltner_mid[i]
        keltner_bearish = price < keltner_mid[i]
        dema_bullish = dema_fast[i] > dema_slow[i]
        dema_bearish = dema_fast[i] < dema_slow[i]
        
        # Strong trend requires both Keltner and DEMA agreement
        strong_bullish = keltner_bullish and dema_bullish
        strong_bearish = keltner_bearish and dema_bearish
        
        # Volume confirmation for entries
        volume_confirmed = vol_ratio >= VOLUME_RATIO
        
        if strong_bullish:
            # Long entry conditions
            stoch_oversold = stoch_k[i] < STOCH_OVERSOLD or stoch_d[i] < STOCH_OVERSOLD
            
            if stoch_oversold and volume_confirmed:
                # Full position on strong pullback with volume
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
            elif stoch_k[i] < 50:
                # Half position on moderate pullback
                signals[i] = SIZE_HALF
                position_side[i] = 1
                entry_price[i] = price
            else:
                # Hold existing long or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    
        elif strong_bearish:
            # Short entry conditions
            stoch_overbought = stoch_k[i] > STOCH_OVERBOUGHT or stoch_d[i] > STOCH_OVERBOUGHT
            
            if stoch_overbought and volume_confirmed:
                # Full short on strong rally with volume
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
            elif stoch_k[i] > 50:
                # Half short on moderate rally
                signals[i] = -SIZE_HALF
                position_side[i] = -1
                entry_price[i] = price
            else:
                # Hold existing short or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
        else:
            # No clear trend - exit positions
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
    
    return signals