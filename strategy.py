#!/usr/bin/env python3
"""
Experiment #124: 4h KAMA + Supertrend + 1d HMA Trend Filter + Volume Confirmation

Hypothesis: Building on #118's success (Sharpe=0.478), this strategy combines:
- KAMA(10,2,30) for adaptive trend following (responds to market efficiency)
- Supertrend(10,3) for volatility-based trend confirmation and stops
- 1d HMA(21) for higher-timeframe trend bias (prevents counter-trend trades)
- Volume ratio filter to confirm breakouts (reduces false signals)
- Asymmetric sizing: 0.35 for strong trend alignment, 0.20 for moderate

Why this might beat #118:
- Supertrend adds volatility-based stop levels that KAMA alone doesn't provide
- Volume confirmation filters false breakouts (major issue in crypto)
- 1d HMA filter prevents entering against major trend (critical for 2022 crash)
- More trades than pure Donchian, fewer than RSI mean-reversion

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete levels
Stoploss: Supertrend levels + 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_supertrend_1d_hma_volume_v1"
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

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - moves fast in trending markets, slow in ranging.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(efficiency_period, n):
        signal = np.abs(close[i] - close[i - efficiency_period])
        noise = np.sum(np.abs(np.diff(close[i-efficiency_period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # First KAMA value = close price
    kama[efficiency_period] = close[efficiency_period]
    
    # Calculate KAMA
    for i in range(efficiency_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_supertrend(high, low, close, period=10, multiplier=3):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    supertrend[:] = np.nan
    direction[:] = np.nan
    
    # Initialize
    supertrend[period] = lower_band[period]
    direction[period] = 1
    
    for i in range(period + 1, n):
        if np.isnan(atr[i]):
            continue
            
        # Calculate new bands
        if close[i-1] > supertrend[i-1]:
            # Previously in uptrend
            lower_band_final = max(lower_band[i], supertrend[i-1])
            if close[i] > lower_band_final:
                supertrend[i] = lower_band_final
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previously in downtrend
            upper_band_final = min(upper_band[i], supertrend[i-1])
            if close[i] < upper_band_final:
                supertrend[i] = upper_band_final
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    kama = calculate_kama(close, 10, 2, 30)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3)
    atr = calculate_atr(high, low, close, 14)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama[i]) or np.isnan(supertrend[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if atr[i] == 0:
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND ===
        # Price above KAMA = bullish, below = bearish
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === SUPERTREND SIGNAL ===
        # direction = 1 means uptrend (price above supertrend)
        # direction = -1 means downtrend (price below supertrend)
        st_bull = st_direction[i] == 1
        st_bear = st_direction[i] == -1
        
        # === VOLUME CONFIRMATION ===
        # Volume must be above average to confirm breakout (relaxed to 1.2x)
        vol_confirmed = volume[i] > 1.2 * vol_ma[i] if not np.isnan(vol_ma[i]) else True
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: 1d bullish + KAMA bull + Supertrend bull + Volume confirmed
        if bull_trend_1d and kama_bull and st_bull and vol_confirmed:
            new_signal = SIZE_STRONG
        # Moderate: 1d bullish + KAMA bull + Supertrend bull
        elif bull_trend_1d and kama_bull and st_bull:
            new_signal = SIZE_BASE
        # Weak: KAMA bull + Supertrend bull (ensure trades on all symbols)
        elif kama_bull and st_bull:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: 1d bearish + KAMA bear + Supertrend bear + Volume confirmed
        if bear_trend_1d and kama_bear and st_bear and vol_confirmed:
            new_signal = -SIZE_STRONG
        # Moderate: 1d bearish + KAMA bear + Supertrend bear
        elif bear_trend_1d and kama_bear and st_bear:
            new_signal = -SIZE_BASE
        # Weak: KAMA bear + Supertrend bear (ensure trades on all symbols)
        elif kama_bear and st_bear:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - Supertrend + ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: max of Supertrend level and 2.5*ATR below highest
            supertrend_stop = supertrend[i] if not np.isnan(supertrend[i]) else 0
            atr_stop = highest_close - 2.5 * atr[i]
            stoploss_price = max(supertrend_stop, atr_stop)
            if close[i] < stoploss_price and stoploss_price > 0:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: min of Supertrend level and 2.5*ATR above lowest
            supertrend_stop = supertrend[i] if not np.isnan(supertrend[i]) else 0
            atr_stop = lowest_close + 2.5 * atr[i]
            if supertrend_stop > 0:
                stoploss_price = min(supertrend_stop, atr_stop)
            else:
                stoploss_price = atr_stop
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals