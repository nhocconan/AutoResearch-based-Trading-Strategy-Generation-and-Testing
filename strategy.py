#!/usr/bin/env python3
"""
Experiment #215: 12h KAMA Adaptive Trend + 1d HMA Filter + Volume + ADX + ATR Stop

Hypothesis: 12h timeframe with KAMA (Kaufman Adaptive Moving Average) captures
multi-day trends while adapting to market noise. KAMA flattens in choppy markets
and trends in directional markets. Combined with 1d HMA for HTF bias, volume
confirmation for breakouts, and ADX filter for trend strength. This should
outperform pure Donchian breakouts which whipsaw in ranges.

Why this might work on 12h:
- KAMA adapts efficiency ratio - fast in trends, slow in ranges
- 1d HMA provides stable higher-timeframe trend bias
- Volume spike (>1.5x avg) confirms breakout validity
- ADX > 15 (not 25) ensures enough trades on 12h timeframe
- ATR 2.5x trailing stop protects against reversals
- Conservative sizing (0.30) controls drawdown

Learning from failures:
- #203 (12h CRSI mean reversion): Sharpe=-0.351 - mean rev fails on crypto
- #209 (12h regime adaptive): Sharpe=-0.740 - too complex, overfitted
- #214 (4h DEMA): Sharpe=0.191 - DEMA too sensitive, KAMA more adaptive
- Trend following works better than mean reversion on crypto
- Need HTF filter to avoid counter-trend trades
- 12h needs lower ADX threshold to generate enough trades

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_1d_hma_volume_adx_atr_v1"
timeframe = "12h"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - fast in trends, slow in ranges.
    Efficiency Ratio determines smoothing constant.
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    # ER = |close - close[period]| / sum(|close[i] - close[i-1]|)
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    # SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Fill initial values
    kama[:period] = close[:period]
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Calculate if volume is spiking above average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio > threshold

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

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
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    kama_10 = calculate_kama(close, period=10, fast=2, slow=30)
    kama_30 = calculate_kama(close, period=30, fast=2, slow=30)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    vol_spike = calculate_volume_spike(volume, 20, 1.5)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(kama_10[i]) or np.isnan(kama_30[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 15 = trending market (lower threshold for 12h to ensure trades)
        trend_strength = adx[i] > 15
        
        # === KAMA ADAPTIVE TREND ===
        # KAMA10 > KAMA30 = bullish adaptive trend
        # KAMA10 < KAMA30 = bearish adaptive trend
        kama_bullish = kama_10[i] > kama_30[i]
        kama_bearish = kama_10[i] < kama_30[i]
        
        # KAMA slope confirmation
        kama_slope_bull = kama_10[i] > kama_10[i-1] if i > 0 else False
        kama_slope_bear = kama_10[i] < kama_10[i-1] if i > 0 else False
        
        # === EMA CONFIRMATION ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === VOLUME CONFIRMATION ===
        # Volume spike confirms breakout validity
        volume_confirmed = vol_spike[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 1d bullish + ADX trending + KAMA bullish + (volume OR EMA bullish)
        # More flexible to ensure enough trades on 12h
        if bull_trend_1d and trend_strength and kama_bullish and kama_slope_bull:
            # Need at least one confirmation (volume OR EMA)
            if volume_confirmed or ema_bullish:
                new_signal = SIZE_BASE
        
        # Short: 1d bearish + ADX trending + KAMA bearish + (volume OR EMA bearish)
        if bear_trend_1d and trend_strength and kama_bearish and kama_slope_bear:
            # Need at least one confirmation (volume OR EMA)
            if volume_confirmed or ema_bearish:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING ===
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0:
            if not in_position or position_side == 0:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else close[i]
                lowest_close = close[i] if position_side < 0 else close[i]
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else close[i]
                lowest_close = close[i] if position_side < 0 else close[i]
            # else: maintaining same position direction, update extremes
            elif position_side > 0 and close[i] > highest_close:
                highest_close = close[i]
            elif position_side < 0 and (lowest_close == 0.0 or close[i] < lowest_close):
                lowest_close = close[i]
        else:
            # No position or exiting
            if in_position and prev_signal != 0.0:
                # Position was active, now exiting
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals