#!/usr/bin/env python3
"""
EXPERIMENT #019 - Supertrend + Stochastic + Volume + ADX Filter + ATR Stop
====================================================================================
Hypothesis: Supertrend(4h) provides cleaner trend signals than Donchian with built-in ATR.
Combined with Stochastic(1h) for precise entry timing, volume confirmation, and ADX
strength filter, this should reduce false signals while capturing strong trends.

Key differences from current best (#017):
- Supertrend(4h, ATR=10, mult=3) instead of Donchian(20) - smoother trend with vol adjustment
- Stochastic(1h, 14,3,3) instead of RSI(14) - better for entry timing in trends
- ADX(14) > 20 filter - only trade when trend has strength
- Volume spike confirmation (>1.5x 20-period avg) - same as #017
- ATR(14) trailing stop at 2.5*ATR - same as #017
- Discrete position sizing (0.0, ±0.25, ±0.35) - same as #017

Why this might beat Sharpe=6.689:
- Supertrend adapts to volatility better than fixed Donchian channels
- Stochastic oversold/overbought in trend direction = higher probability entries
- ADX filter avoids weak trends that cause whipsaws
- Multi-timeframe logic proven in #017 (4h trend + 1h entries)
- Conservative position sizing (max 0.35) controls drawdown
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_stoch_volume_adx_v1"
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
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band = mid + multiplier * atr[i]
        lower_band = mid - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            if close[i] > supertrend[i - 1]:
                supertrend[i] = lower_band
                direction[i] = 1
            else:
                supertrend[i] = upper_band
                direction[i] = -1
    
    return supertrend, direction


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
            k[i] = 100
    
    # Calculate %D (SMA of %K)
    for i in range(k_period - 1 + d_period - 1, n):
        d[i] = np.mean(k[i - d_period + 1:i + 1])
    
    return k, d


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
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    atr[period - 1] = np.mean(tr[1:period])
    plus_di[period - 1] = 100 * np.mean(plus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    minus_di[period - 1] = 100 * np.mean(minus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        plus_di[i] = 100 * ((plus_di[i - 1] * (period - 1) + 100 * plus_dm[i] / atr[i - 1]) / period) if atr[i] > 0 else 0
        minus_di[i] = 100 * ((minus_di[i - 1] * (period - 1) + 100 * minus_dm[i] / atr[i - 1]) / period) if atr[i] > 0 else 0
    
    dx = np.zeros(n)
    for i in range(period, n):
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    # ADX = SMA of DX
    adx = np.zeros(n)
    adx[2 * period - 1] = np.mean(dx[period:2 * period])
    
    for i in range(2 * period, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above moving average"""
    n = len(volume)
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = np.zeros(n)
    
    mask = vol_ma > 0
    spike[mask] = (volume[mask] / vol_ma[mask]) > threshold
    
    return spike


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    stoch_k_1h, stoch_d_1h = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    atr_1h = calculate_atr(high, low, close, period=14)
    adx_1h = calculate_adx(high, low, close, period=14)
    volume_spike_1h = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    # 4h Supertrend for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Calculate 4h Supertrend
    supertrend_4h, direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(direction_4h):
            trend_1h[i] = direction_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position with volume confirmation
    SIZE_HALF = 0.25   # Reduced position without volume
    
    # Stochastic thresholds for entries
    STOCH_LONG_ENTRY = 40   # Enter long when Stoch crosses above 40 in uptrend
    STOCH_SHORT_ENTRY = 60  # Enter short when Stoch crosses below 60 in downtrend
    
    # ADX threshold for trend strength
    ADX_MIN = 20
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 20, 14, 28)  # Wait for all indicators (ADX needs 2*period)
    
    # Track entry prices for trailing stop logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    prev_stoch_k = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(stoch_k_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        stoch_k = stoch_k_1h[i]
        stoch_d = stoch_d_1h[i]
        adx = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        vol_spike = volume_spike_1h[i]
        
        # Store previous stoch for cross detection
        if i > 0:
            prev_stoch_k[i] = stoch_k_1h[i - 1]
        else:
            prev_stoch_k[i] = stoch_k
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
        
        # ADX filter - only trade when trend has strength
        if adx < ADX_MIN:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Determine position size based on volume confirmation
        position_size = SIZE_FULL if vol_spike else SIZE_HALF
        
        if trend == 1:  # 4h uptrend
            # Stochastic entry in uptrend (cross above 40 from below)
            if prev_stoch_k[i] < STOCH_LONG_ENTRY and stoch_k >= STOCH_LONG_ENTRY:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # Stochastic entry in downtrend (cross below 60 from above)
            if prev_stoch_k[i] > STOCH_SHORT_ENTRY and stoch_k <= STOCH_SHORT_ENTRY:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals