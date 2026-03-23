#!/usr/bin/env python3
"""
Experiment #971: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + RSI Pullback + ADX Regime

Hypothesis: After 664+ failed strategies, KAMA (Kaufman Adaptive Moving Average) provides
superior trend following in crypto because it adapts to volatility - fast during trends,
slow during ranges. Combined with RSI pullback entries and ADX regime filter, this should
work across ALL symbols (BTC/ETH/SOL) in both bull and bear markets.

Why this differs from failed attempts:
1. KAMA instead of HMA/EMA - adapts efficiency ratio to market conditions
2. ADX for regime (not Choppiness which failed in #960, #961, #966)
3. SIMPLER entry logic - fewer confluence filters = more trades (avoid 0-trade failure)
4. 1d HMA for trend bias, 1w HMA for macro regime (proven in #964 with Sharpe=0.177)
5. RELAXED RSI thresholds (35/65 not 30/70) to ensure trades trigger

Key mechanics:
- KAMA(10,2,30): Fast SC=2/(2+1), Slow SC=2/(30+1), ER based on 10-bar net change vs sum ABS changes
- Long: 1d HMA bullish + KAMA slope up + RSI 35-50 pullback + ADX > 20
- Short: 1d HMA bearish + KAMA slope down + RSI 50-65 pullback + ADX > 20
- Exit: RSI > 70 (long) or < 30 (short) OR KAMA slope flips
- Stoploss: 2.5x ATR trailing stop via signal→0

Timeframe: 4h (target 30-50 trades/year, fee drag 1-2.5%)
Position sizing: 0.25 base, 0.30 strong signal (discrete levels minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_pullback_1d1w_adx_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing constant based on market efficiency (trend vs noise).
    Fast SC = 2/(fast_period+1), Slow SC = 2/(slow_period+1)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(er_period - 1, n):
        signal = np.abs(close[i] - close[i - er_period + 1])
        noise = np.sum(np.abs(np.diff(close[i - er_period + 1:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period - 1] = close[er_period - 1]
    
    for i in range(er_period, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
            continue
        
        # Adaptive smoothing constant
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
    """Average Directional Index - measures trend strength."""
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
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_dm_s / (tr_s + 1e-10)
        minus_di = 100 * minus_dm_s / (tr_s + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    adx[period*2-1:] = adx_raw[period*2-1:]
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

def calculate_hma(series, period):
    """Hull Moving Average for HTF trend."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_4h = calculate_rsi(close, period=14)
    adx_4h = calculate_adx(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(adx_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # Calculate KAMA slope (rate of change over 3 bars)
        kama_slope = 0.0
        if i >= 3 and not np.isnan(kama_4h[i-3]):
            kama_slope = (kama_4h[i] - kama_4h[i-3]) / (kama_4h[i-3] + 1e-10)
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_4h[i] > 20
        very_strong_trend = adx_4h[i] > 30
        
        # === RSI PULLBACK ZONES ===
        rsi_pullback_long = 35 <= rsi_4h[i] <= 50
        rsi_pullback_short = 50 <= rsi_4h[i] <= 65
        rsi_extreme_long = rsi_4h[i] < 35
        rsi_extreme_short = rsi_4h[i] > 65
        rsi_overbought = rsi_4h[i] > 70
        rsi_oversold = rsi_4h[i] < 30
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = kama_slope > 0.001
        kama_bearish = kama_slope < -0.001
        price_above_kama = close[i] > kama_4h[i]
        price_below_kama = close[i] < kama_4h[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: 1d trend bullish + KAMA slope up + RSI pullback + ADX confirms trend
        if trend_1d_bullish and kama_bullish and rsi_pullback_long and strong_trend:
            desired_signal = BASE_SIZE
        
        # Strong: Add macro bullish + price above KAMA
        if trend_1d_bullish and macro_bull and kama_bullish and price_above_kama:
            if rsi_pullback_long or rsi_extreme_long:
                desired_signal = STRONG_SIZE
        
        # Secondary: RSI extreme oversold in bullish macro (mean reversion)
        if macro_bull and rsi_extreme_long and strong_trend:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: 1d trend bearish + KAMA slope down + RSI pullback + ADX confirms trend
        if trend_1d_bearish and kama_bearish and rsi_pullback_short and strong_trend:
            desired_signal = -BASE_SIZE
        
        # Strong: Add macro bearish + price below KAMA
        if trend_1d_bearish and macro_bear and kama_bearish and price_below_kama:
            if rsi_pullback_short or rsi_extreme_short:
                desired_signal = -STRONG_SIZE
        
        # Secondary: RSI extreme overbought in bearish macro (mean reversion)
        if macro_bear and rsi_extreme_short and strong_trend:
            desired_signal = -BASE_SIZE
        
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
                # Hold long if 1d trend and KAMA still bullish
                if trend_1d_bullish and kama_bullish and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d trend and KAMA still bearish
                if trend_1d_bearish and kama_bearish and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if RSI overbought OR KAMA flips bearish
            if rsi_overbought or (kama_bearish and price_below_kama):
                desired_signal = 0.0
            # Exit if macro + 1d trend both flip bearish
            if macro_bear and trend_1d_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if RSI oversold OR KAMA flips bullish
            if rsi_oversold or (kama_bullish and price_above_kama):
                desired_signal = 0.0
            # Exit if macro + 1d trend both flip bullish
            if macro_bull and trend_1d_bullish:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = STRONG_SIZE if desired_signal >= STRONG_SIZE else BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -STRONG_SIZE if desired_signal <= -STRONG_SIZE else -BASE_SIZE
        
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