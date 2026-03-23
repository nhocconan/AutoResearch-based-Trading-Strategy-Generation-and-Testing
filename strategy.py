#!/usr/bin/env python3
"""
Experiment #849: 4h Primary + 1d HTF — KAMA Adaptive Trend + ADX + RSI Pullback

Hypothesis: After 585+ failed strategies, the key insight is that 4h timeframe
needs ADAPTIVE trend following (KAMA) + momentum confirmation (ADX) to work
in both bull and bear markets. Pure Fisher/Choppiness strategies failed on 1d.

Strategy design:
1. 4h Primary timeframe (target 25-45 trades/year)
2. 1d HMA(21) for long-term bias only (call ONCE before loop)
3. 4h KAMA(10) for adaptive trend following (adjusts to volatility)
4. 4h ADX(14) for trend strength confirmation (>25 = trending)
5. 4h RSI(14) pullback entries in trend direction
6. 4h ATR(14) for trailing stop (2.5x)
7. Dual logic: trend-follow when ADX>25, mean-revert when ADX<20
8. Position sizing: 0.25-0.30 discrete levels

Why KAMA:
- Adapts smoothing based on market efficiency ratio
- Less whipsaw than EMA/HMA in ranging markets
- Proven edge in crypto perpetual futures

Why ADX filter:
- ADX>25 = strong trend (follow direction)
- ADX<20 = weak/ranging (mean revert at extremes)
- Avoids trading in choppy conditions

Target: Sharpe > 0.612, trades >= 20 train, >= 5 test, ALL symbols positive
Timeframe: 4h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi_pullback_1d_atr_v1"
timeframe = "4h"
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

def calculate_kama(close, period=10):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency ratio.
    Less whipsaw in ranging markets.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + 1:
        return kama
    
    # Efficiency Ratio parameters
    er_period = 10
    fast_sc = 2.0 / (2 + 1)
    slow_sc = 2.0 / (2 + 30)
    
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        # Calculate change and volatility
        change = np.abs(close[i] - close[i - er_period]) if i >= er_period else np.abs(close[i] - close[period - 1])
        
        volatility = 0.0
        for j in range(i - er_period + 1, i + 1):
            if j > 0:
                volatility += np.abs(close[j] - close[j - 1])
        
        # Efficiency Ratio
        er = change / (volatility + 1e-10) if volatility > 1e-10 else 0.0
        er = np.clip(er, 0, 1)
        
        # Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
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
    Average Directional Index - measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = weak/ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx, plus_di, minus_di
    
    # Calculate True Range and DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR and DM
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_plus_dm = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_minus_dm = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di_arr = 100 * smoothed_plus_dm / (atr + 1e-10)
        minus_di_arr = 100 * smoothed_minus_dm / (atr + 1e-10)
    
    plus_di[:] = plus_di_arr
    minus_di[:] = minus_di_arr
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period * 2, n):
        di_sum = plus_di_arr[i] + minus_di_arr[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(plus_di_arr[i] - minus_di_arr[i]) / di_sum
    
    # Smooth DX to get ADX
    adx_arr = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:] = adx_arr
    
    return adx, plus_di, minus_di

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

def calculate_donchian(high, low, period=20):
    """Donchian Channels — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10)
    rsi_4h = calculate_rsi(close, period=14)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1d HMA for long-term trend bias
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
        if np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(adx_4h[i]) or np.isnan(plus_di_4h[i]) or np.isnan(minus_di_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SECULAR TREND FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx_4h[i] > 25
        weak_trend = adx_4h[i] < 20
        moderate_trend = 20 <= adx_4h[i] <= 25
        
        # === DI DIRECTION ===
        di_bullish = plus_di_4h[i] > minus_di_4h[i]
        di_bearish = plus_di_4h[i] < minus_di_4h[i]
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        rsi_extreme_oversold = rsi_4h[i] < 25
        rsi_extreme_overbought = rsi_4h[i] > 75
        rsi_pullback_long = 35 <= rsi_4h[i] < 50
        rsi_pullback_short = 50 < rsi_4h[i] <= 65
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        desired_signal = 0.0
        
        # === STRONG TREND REGIME (ADX > 25) — Trend Following ===
        if strong_trend:
            # Long: All trend signals aligned + RSI not overbought
            if trend_1d_bullish and kama_bullish and di_bullish and rsi_4h[i] < 65:
                # Entry on pullback or breakout
                if rsi_pullback_long or donchian_breakout_long:
                    desired_signal = BASE_SIZE
                elif above_sma200 and rsi_4h[i] < 55:
                    desired_signal = REDUCED_SIZE
            
            # Short: All trend signals aligned + RSI not oversold
            if trend_1d_bearish and kama_bearish and di_bearish and rsi_4h[i] > 35:
                # Entry on pullback or breakout
                if rsi_pullback_short or donchian_breakout_short:
                    desired_signal = -BASE_SIZE
                elif below_sma200 and rsi_4h[i] > 45:
                    desired_signal = -REDUCED_SIZE
        
        # === WEAK TREND REGIME (ADX < 20) — Mean Reversion ===
        elif weak_trend:
            # Long: Extreme oversold + any trend support
            if rsi_extreme_oversold and (trend_1d_bullish or above_sma200):
                desired_signal = REDUCED_SIZE
            
            # Short: Extreme overbought + any trend resistance
            if rsi_extreme_overbought and (trend_1d_bearish or below_sma200):
                desired_signal = -REDUCED_SIZE
            
            # Moderate mean reversion with KAMA support
            if rsi_oversold and close[i] > kama_4h[i] and trend_1d_bullish:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if rsi_overbought and close[i] < kama_4h[i] and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === MODERATE TREND REGIME (20 <= ADX <= 25) — Hybrid ===
        else:
            # Conservative trend following with RSI filter
            if trend_1d_bullish and kama_bullish and rsi_pullback_long:
                desired_signal = REDUCED_SIZE
            
            if trend_1d_bearish and kama_bearish and rsi_pullback_short:
                desired_signal = -REDUCED_SIZE
            
            # Breakout confirmation
            if donchian_breakout_long and trend_1d_bullish:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if donchian_breakout_short and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === ENSURE TRADES ON ALL SYMBOLS (fallback for low volatility periods) ===
        if desired_signal == 0.0:
            # Very simple fallback to guarantee trades
            if rsi_extreme_oversold and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            if rsi_extreme_overbought and trend_1d_bearish:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and RSI not overbought
                if (trend_1d_bullish or kama_bullish) and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and RSI not oversold
                if (trend_1d_bearish or kama_bearish) and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses
            if trend_1d_bearish and kama_bearish and rsi_4h[i] > 65:
                desired_signal = 0.0
            # Exit if RSI extremely overbought
            if rsi_4h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses
            if trend_1d_bullish and kama_bullish and rsi_4h[i] < 35:
                desired_signal = 0.0
            # Exit if RSI extremely oversold
            if rsi_4h[i] < 20:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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