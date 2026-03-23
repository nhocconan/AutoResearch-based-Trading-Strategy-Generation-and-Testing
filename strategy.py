#!/usr/bin/env python3
"""
Experiment #892: 12h Primary + 1d HTF — Simplified KAMA Trend + ADX Regime + RSI Pullback

Hypothesis: After 600+ failed strategies, the key is SIMPLER logic with fewer conflicting filters.
Previous 12h strategies failed due to overly complex regime detection (3+ states) and too many
confluence requirements that never all aligned. This strategy uses:

1. 12h Primary TF: Target 25-40 trades/year (lower fee drag)
2. 1d HMA(21) for trend bias only (single HTF filter, not 1w + 1d)
3. KAMA(14) for adaptive trend following (better than EMA in chop)
4. ADX(14) for regime: ADX>25=trend, ADX<20=range (binary, not 3 states)
5. RSI(14) for entries: pullback in trend, extremes in range
6. ATR(14) trailing stop (2.5x) for risk management

Why this should work:
- SIMPLER logic = more trades across ALL symbols (BTC/ETH/SOL)
- KAMA adapts to volatility better than HMA/EMA
- Binary regime (trend/range) is more reliable than 3-state choppiness
- Single HTF filter (1d) reduces filter conflict
- Relaxed RSI thresholds (30/70 not 20/80) ensure entries trigger
- Conservative sizing (0.25-0.30) controls drawdown

Critical improvements from failed experiments:
- REMOVED 1w HMA filter (was causing 0 trades on ETH)
- REMOVED Choppiness Index (unreliable, caused whipsaw)
- REMOVED Donchian breakout (too many false signals)
- REMOVED CRSI (complex, no edge over simple RSI)
- ADX binary regime is proven in literature (Wilder's original)
- KAMA efficiency ratio adapts to market conditions automatically

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_regime_1d_hma_rsi_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise via Efficiency Ratio.
    ER = |net change| / sum of absolute changes
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        net_change = np.abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 0:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate KAMA
    kama[period - 1] = close[period - 1]
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX).
    ADX > 25 = trending, ADX < 20 = ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed values (Wilder's smoothing)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI calculations
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_di / (atr + 1e-10)
        minus_di = 100 * minus_di / (atr + 1e-10)
    
    # DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, period=14)
    rsi_12h = calculate_rsi(close, period=14)
    adx_12h = calculate_adx(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(kama_12h[i]) or np.isnan(rsi_12h[i]) or np.isnan(adx_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        
        # === TREND BIAS (1d HTF HMA21) ===
        trend_bullish = close[i] > hma_1d_aligned[i]
        trend_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (ADX binary) ===
        trending_regime = adx_12h[i] > 25
        ranging_regime = adx_12h[i] < 20
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = close[i] > kama_12h[i]
        kama_bearish = close[i] < kama_12h[i]
        
        # === RSI SIGNALS (Relaxed: 30/70 for entries) ===
        rsi_oversold = rsi_12h[i] < 35
        rsi_overbought = rsi_12h[i] > 65
        rsi_extreme_oversold = rsi_12h[i] < 25
        rsi_extreme_overbought = rsi_12h[i] > 75
        
        # === SMA FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (ADX > 25) — Trend Following with Pullback ===
        if trending_regime:
            # Long: Bullish trend + RSI pullback
            if trend_bullish and kama_bullish and above_sma50:
                if rsi_oversold:
                    desired_signal = BASE_SIZE
                elif rsi_12h[i] < 45 and above_sma200:
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + RSI bounce
            if trend_bearish and kama_bearish and below_sma50:
                if rsi_overbought:
                    desired_signal = -BASE_SIZE
                elif rsi_12h[i] > 55 and below_sma200:
                    desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME (ADX < 20) — Mean Reversion ===
        elif ranging_regime:
            # Long: RSI extreme oversold
            if rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            elif rsi_oversold and above_sma200:
                desired_signal = REDUCED_SIZE
            
            # Short: RSI extreme overbought
            if rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            elif rsi_overbought and below_sma200:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (20 <= ADX <= 25) — Conservative ===
        else:
            # Only enter on extreme RSI with trend confirmation
            if rsi_extreme_oversold and (trend_bullish or above_sma50):
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and (trend_bearish or below_sma50):
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and RSI not overbought
                if trend_bullish and rsi_12h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and RSI not oversold
                if trend_bearish and rsi_12h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses + RSI overbought
            if trend_bearish and rsi_12h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses + RSI oversold
            if trend_bullish and rsi_12h[i] < 30:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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
        
        signals[i] = desired_signal
    
    return signals