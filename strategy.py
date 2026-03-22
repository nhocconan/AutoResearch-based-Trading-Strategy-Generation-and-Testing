#!/usr/bin/env python3
"""
Experiment #018: 1d Supertrend + HMA Crossover with 1w Regime Filter
Hypothesis: Daily timeframe reduces noise and fee impact. Supertrend provides clear trend direction
with ATR-based stops built-in. HMA crossover confirms momentum. 1w HMA acts as simple regime filter
(bull/bear bias) without over-complicating entries. Volume spike confirms breakout validity.
This is SIMPLER than previous attempts - fewer filters = more trades generated.
Position sizing: 0.25 base, 0.30 in bull regime, 0.20 in bear regime.
Stoploss: 2.5*ATR trailing stop (in addition to Supertrend flip).
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_supertrend_hma_1w_regime_v1"
timeframe = "1d"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR-based stops.
    Returns: supertrend values, direction (1=up, -1=down)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)
    direction[:] = np.nan
    
    hl2 = (high + low) / 2.0
    
    # Upper and lower bands
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize
    supertrend[period] = upper_band[period]
    direction[period] = -1  # Start bearish
    
    for i in range(period + 1, n):
        if direction[i - 1] == 1:
            # Previous was bullish
            if lower_band[i] < supertrend[i - 1]:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = supertrend[i - 1]
        else:
            # Previous was bearish
            if upper_band[i] > supertrend[i - 1]:
                supertrend[i] = upper_band[i]
            else:
                supertrend[i] = supertrend[i - 1]
        
        # Update direction
        if close[i] > supertrend[i]:
            direction[i] = 1
        else:
            direction[i] = -1
    
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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / vol_ma
    ratio[np.isnan(ratio)] = 1.0
    ratio[np.isinf(ratio)] = 1.0
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # HMA for crossover signals
    hma_fast = calculate_hma(close, 16)
    hma_slow = calculate_hma(close, 48)
    
    # Volume confirmation
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Additional trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - asymmetric based on regime
    SIZE_BULL = 0.30  # Larger in bull regime
    SIZE_BEAR = 0.20  # Smaller in bear regime (risk management)
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            continue
        
        # 1w regime bias (HTF) - determines which direction to favor
        bull_regime = close[i] > hma_1w_aligned[i]
        bear_regime = close[i] < hma_1w_aligned[i]
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # HMA crossover
        hma_golden = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_death = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        hma_bullish = hma_fast[i] > hma_slow[i]
        hma_bearish = hma_fast[i] < hma_slow[i]
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.1  # 10% above average
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and ema_50[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and ema_50[i] < ema_200[i]
        
        # Select position size based on regime
        current_size = SIZE_BULL if bull_regime else SIZE_BEAR
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Supertrend bullish + HMA bullish + bull regime
        if st_bullish and hma_bullish and bull_regime:
            new_signal = current_size
        # Secondary: Supertrend flip to bullish + volume confirmation
        elif st_direction[i] == 1 and st_direction[i-1] == -1 and volume_confirmed:
            new_signal = current_size
        # Tertiary: HMA golden cross + bull regime
        elif hma_golden and bull_regime:
            new_signal = current_size
        
        # === SHORT ENTRY ===
        # Primary: Supertrend bearish + HMA bearish + bear regime
        if st_bearish and hma_bearish and bear_regime:
            new_signal = -current_size
        # Secondary: Supertrend flip to bearish + volume confirmation
        elif st_direction[i] == -1 and st_direction[i-1] == 1 and volume_confirmed:
            new_signal = -current_size
        # Tertiary: HMA death cross + bear regime
        elif hma_death and bear_regime:
            new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Supertrend flip acts as primary stop
        if position_side > 0 and st_bearish:
            new_signal = 0.0
        
        if position_side < 0 and st_bullish:
            new_signal = 0.0
        
        # ATR trailing stop (secondary)
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals