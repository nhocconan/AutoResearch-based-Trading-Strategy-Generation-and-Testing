#!/usr/bin/env python3
"""
Experiment #180: 1d KAMA Trend + 1w HMA Filter + ADX + Donchian Breakout + ATR Stop

Hypothesis: Daily timeframe with weekly trend filter captures major crypto cycles
while filtering noise. KAMA adapts to volatility (fast in trends, slow in ranges).
Donchian breakouts (20-period) capture momentum moves. ADX > 20 ensures trending
market. 1w HMA provides stable higher-timeframe bias. ATR trailing stop protects
against reversals. This should generate fewer but higher-quality trades than 4h/12h.

Why 1d might work:
- 1d bars filter intraday noise and fake breakouts
- Weekly HMA = very stable trend bias (only changes on major shifts)
- KAMA adapts to volatility better than EMA/SMA
- Donchian 20 on 1d = ~20 day breakout (Turtle Trading style)
- ADX > 20 (not 25+) ensures enough trades on daily timeframe
- Conservative sizing (0.30) with 3*ATR stop controls drawdown

Learning from failures:
- #168 (1d KAMA + 1w HMA): Sharpe=-0.859 - BB squeeze + RSI too restrictive
- #174 (1d KAMA + 1w HMA + RSI): Sharpe=-0.004 - almost broke even, need breakout logic
- #178 (4h Donchian): Sharpe=-0.989 - 4h too noisy, 1d should be better
- #179 (12h Donchian): Sharpe=-0.319 - close to breaking even, 1d + 1w filter should improve

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 3.0 * ATR(14) trailing (wider for 1d)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_1w_hma_donchian_adx_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman's Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = 0
    
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    volatility[:er_period] = change[:er_period]
    volatility = np.where(volatility == 0, 1e-10, volatility)
    
    er = change / volatility
    
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    upper[:period-1] = upper[period-1]
    lower[:period-1] = lower[period-1]
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    kama = calculate_kama(close, 10, 2, 30)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.30
    
    # Track for stoploss
    highest_close = 0.0
    lowest_close = 0.0
    in_position = False
    position_side = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # === TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # === ADX FILTER ===
        trend_strength = adx[i] > 20
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        if bull_trend_1w and kama_bullish and trend_strength and breakout_long:
            new_signal = SIZE_BASE
        
        if bear_trend_1w and kama_bearish and trend_strength and breakout_short:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        if in_position:
            if position_side > 0:
                highest_close = max(highest_close, close[i])
                stoploss_price = highest_close - 3.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0:
                    lowest_close = close[i]
                else:
                    lowest_close = min(lowest_close, close[i])
                stoploss_price = lowest_close + 3.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals