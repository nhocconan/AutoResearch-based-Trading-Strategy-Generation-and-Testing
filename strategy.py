#!/usr/bin/env python3
"""
Experiment #210: 1d KAMA Adaptive Trend + 1w HMA Filter + Volume Spike + ATR Stop

Hypothesis: Daily timeframe with Kaufman Adaptive Moving Average (KAMA) captures
trends while adapting to market volatility. KAMA flattens in choppy markets and
tracks price in trends. Combined with 1w HMA for ultra-stable higher-timeframe
bias, volume spike confirmation (1.5x average), and RSI momentum filter, this
should generate quality signals with enough trade count on 1d.

Why this might work on 1d:
- KAMA adapts to market efficiency ratio (ER) - less whipsaw than EMA
- 1w HMA = very stable trend bias (only changes on major regime shifts)
- Volume spike = confirms breakout validity (avoids fake breakouts)
- RSI 45-55 filter = ensures momentum without being too strict
- Lower ADX threshold (15) = ensures sufficient trades on daily bars
- Conservative sizing (0.30) controls drawdown in 2022-style crashes

Learning from failures:
- #198 (1d KAMA + 1w HMA): Sharpe=0.028 - worked but barely, needs volume filter
- #204 (1d Donchian + 1w HMA): Sharpe=-0.136 - Donchian too slow on 1d
- Mean reversion fails on crypto, trend-following with filters works
- Need volume confirmation to avoid false breakouts on low-volume days
- ADX threshold must be low (15-20) on 1d to ensure ≥10 trades

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_1w_hma_volume_rsi_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    # Calculate True Range and Directional Movement
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
    
    # Smooth with Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise using Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    SC (Smoothing Constant) = [ER * (fast_SC - slow_SC) + slow_SC]^2
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    net_change = np.abs(close_s.diff(er_period))
    sum_changes = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum()
    
    # Avoid division by zero
    er = net_change / (sum_changes + 1e-10)
    er = er.fillna(0)
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er.values * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """
    Calculate volume spike indicator.
    Returns 1 if current volume > threshold * average volume, else 0.
    """
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    spike = (vol_ratio > threshold).astype(float)
    return spike.values

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
    adx = calculate_adx(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=15)
    rsi = calculate_rsi(close, 14)
    vol_spike = calculate_volume_spike(volume, 20, 1.5)
    
    # Calculate EMA for additional confirmation
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
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
        
        if np.isnan(adx[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = ultra-stable higher timeframe trend bias
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 15 = trending market (lower threshold for 1d to ensure trades)
        trend_strength = adx[i] > 15
        
        # === KAMA ADAPTIVE TREND ===
        # Fast KAMA > Slow KAMA = bullish adaptive trend
        # Fast KAMA < Slow KAMA = bearish adaptive trend
        kama_bullish = kama_fast[i] > kama[i]
        kama_bearish = kama_fast[i] < kama[i]
        
        # === RSI MOMENTUM ===
        # RSI > 45 = bullish momentum (not too strict)
        # RSI < 55 = bearish momentum (not too strict)
        rsi_bullish = rsi[i] > 45
        rsi_bearish = rsi[i] < 55
        
        # === VOLUME CONFIRMATION ===
        # Volume spike confirms breakout validity
        volume_confirmed = vol_spike[i] > 0
        
        # === EMA STRUCTURE ===
        # EMA21 > EMA50 = bullish trend structure
        # EMA21 < EMA50 = bearish trend structure
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 1w bullish + ADX trending + KAMA bullish + (RSI OR EMA) + volume preferred
        # Using flexible conditions to ensure enough trades on 1d
        if bull_trend_1w and trend_strength and kama_bullish:
            # Need momentum confirmation (RSI or EMA structure)
            if rsi_bullish or ema_bullish:
                # Volume spike increases confidence but not required
                if volume_confirmed:
                    new_signal = SIZE_BASE
                else:
                    # Enter with smaller size without volume confirmation
                    new_signal = SIZE_BASE * 0.67
        
        # Short: 1w bearish + ADX trending + KAMA bearish + (RSI OR EMA) + volume preferred
        if bear_trend_1w and trend_strength and kama_bearish:
            # Need momentum confirmation (RSI or EMA structure)
            if rsi_bearish or ema_bearish:
                # Volume spike increases confidence but not required
                if volume_confirmed:
                    new_signal = -SIZE_BASE
                else:
                    # Enter with smaller size without volume confirmation
                    new_signal = -SIZE_BASE * 0.67
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
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
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals