#!/usr/bin/env python3
"""
Experiment #243: 6h Primary + 1d/1w HTF — Keltner Channel + ADX Regime

Hypothesis: 6h timeframe with Keltner Channel mean-reversion in low-ADX regimes
and trend-following in high-ADX regimes. Previous 6h attempts used Fisher, funding,
weekly pivots - all failed. This tries Keltner Channels (EMA + ATR bands) which
have different dynamics than Bollinger Bands.

Regime Detection:
- ADX(14) < 20 = range-bound → mean revert at Keltner bands
- ADX(14) > 25 = trending → trade breakouts with HMA confirmation

Keltner Channels:
- Upper = EMA(20) + 2.0 * ATR(14)
- Lower = EMA(20) - 2.0 * ATR(14)
- Long when price touches lower band + ADX low + HTF bull
- Short when price touches upper band + ADX low + HTF bear

1d HTF: HMA(50) for intermediate trend bias
1w HTF: HMA(21) for secular trend filter (only trade in direction)

Position sizing: 0.25 base, 0.30 for strong confluence
Stoploss: 2.5x ATR trailing

Target: Sharpe>0.399 (beat current 6h best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_keltner_adx_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with EMA
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_keltner(high, low, close, ema_period=20, atr_period=14, atr_mult=2.0):
    """
    Keltner Channels - EMA +/- ATR multiplier
    Upper = EMA(20) + 2.0 * ATR(14)
    Lower = EMA(20) - 2.0 * ATR(14)
    """
    n = len(close)
    ema = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(max(ema_period, atr_period), n):
        if not np.isnan(ema[i]) and not np.isnan(atr[i]):
            upper[i] = ema[i] + atr_mult * atr[i]
            lower[i] = ema[i] - atr_mult * atr[i]
    
    return upper, lower, ema

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for intermediate trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for secular trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    keltner_upper, keltner_lower, keltner_ema = calculate_keltner(high, low, close, ema_period=20, atr_period=14, atr_mult=2.0)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # 25% base position size
    SIZE_STRONG = 0.30  # 30% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Start after all indicators are ready (need ~300 bars for 1w alignment)
    start_idx = 300
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(adx[i]) or np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 6h HMA TREND ===
        hma_6h_bull = close[i] > hma_6h[i]
        hma_6h_bear = close[i] < hma_6h[i]
        
        # === REGIME DETECTION (ADX) ===
        is_ranging = adx[i] < 20.0  # Low ADX = range-bound
        is_trending = adx[i] > 25.0  # High ADX = trending
        # 20-25 is transition zone
        
        # === KELTNER CHANNEL POSITION ===
        near_lower_band = close[i] <= keltner_lower[i] * 1.002  # Within 0.2% of lower band
        near_upper_band = close[i] >= keltner_upper[i] * 0.998  # Within 0.2% of upper band
        below_lower_band = close[i] < keltner_lower[i]
        above_upper_band = close[i] > keltner_upper[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = not np.isnan(rsi[i]) and rsi[i] < 35.0
        rsi_overbought = not np.isnan(rsi[i]) and rsi[i] > 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: RANGING (ADX < 20) - Mean reversion at Keltner bands
        if is_ranging:
            # Long: Price at/below lower Keltner + RSI oversold + HTF not strongly bear
            if near_lower_band and rsi_oversold and not (htf_1d_bear and htf_1w_bear):
                desired_signal = SIZE_BASE
            
            # Short: Price at/above upper Keltner + RSI overbought + HTF not strongly bull
            elif near_upper_band and rsi_overbought and not (htf_1d_bull and htf_1w_bull):
                desired_signal = -SIZE_BASE
        
        # REGIME 2: TRENDING (ADX > 25) - Trend following with Keltner breakout
        elif is_trending:
            # Long: Price breaks above upper Keltner + HMA bull + 1d HTF bull
            if above_upper_band and hma_6h_bull and htf_1d_bull:
                desired_signal = SIZE_STRONG
            
            # Short: Price breaks below lower Keltner + HMA bear + 1d HTF bear
            elif below_lower_band and hma_6h_bear and htf_1d_bear:
                desired_signal = -SIZE_STRONG
        
        # REGIME 3: TRANSITION (ADX 20-25) - Require stronger HTF alignment
        else:
            # Long: Keltner lower + HMA bull + 1d bull + 1w bull
            if near_lower_band and hma_6h_bull and htf_1d_bull and htf_1w_bull:
                desired_signal = SIZE_BASE * 0.8
            
            # Short: Keltner upper + HMA bear + 1d bear + 1w bear
            elif near_upper_band and hma_6h_bear and htf_1d_bear and htf_1w_bear:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals