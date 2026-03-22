#!/usr/bin/env python3
"""
Experiment #066: 1d Dual Regime with 1w HMA Trend + Adaptive Entry
Hypothesis: Daily timeframe captures major trends while avoiding noise. Using 1w HMA as 
ultimate trend filter, combined with adaptive entry logic that switches between trend-following
(ADX>25) and mean-reversion (ADX<20) based on market regime. This should work in both
bull (2021) and bear/range (2022, 2025) markets.
Key insight: Most failed strategies used too many filters. This uses fewer, looser conditions
to ensure 10+ trades per symbol while maintaining edge through regime detection.
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.35 strong trend, discrete levels to minimize fee churn.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_1w_hma_adaptive_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    change = np.abs(close - np.roll(close, er_period))
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    
    sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1)) ** 2
    
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    rsi = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    hma_21 = calculate_hma(close, 21)
    hma_10 = calculate_hma(close, 10)
    kama = calculate_kama(close, 10, 2, 30)
    bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = long-term trend bias (most important filter)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # 1d HMA crossover = short-term momentum
        hma_bullish = hma_10[i] > hma_21[i] if not np.isnan(hma_10[i]) else False
        hma_bearish = hma_10[i] < hma_21[i] if not np.isnan(hma_10[i]) else False
        
        # EMA alignment
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # Price vs SMA200
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # KAMA trend
        kama_bullish = close[i] > kama[i] if not np.isnan(kama[i]) else False
        kama_bearish = close[i] < kama[i] if not np.isnan(kama[i]) else False
        
        # === TREND STRENGTH / REGIME ===
        trending_regime = adx[i] > 20  # Looser threshold for more trades
        strong_trend = adx[i] > 28
        ranging_regime = adx[i] < 18
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # Bollinger position
        near_bb_lower = close[i] <= bb_lower[i] * 1.02 if not np.isnan(bb_lower[i]) else False
        near_bb_upper = close[i] >= bb_upper[i] * 0.98 if not np.isnan(bb_upper[i]) else False
        
        # RSI conditions (looser for more trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral = 35 <= rsi[i] <= 65
        rsi_7_oversold = rsi_7[i] < 30
        rsi_7_overbought = rsi_7[i] > 70
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Trend following in trending regime (primary)
        if trending_regime and bull_trend_1w:
            if hma_bullish and di_bullish:
                if strong_trend:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
        
        # Path 2: EMA crossover + trend bias
        if bull_trend_1w and ema_bullish:
            if rsi[i] > 40 and rsi[i] < 70:
                new_signal = SIZE_BASE
        
        # Path 3: Mean reversion in ranging regime
        if ranging_regime:
            if rsi_oversold and near_bb_lower:
                new_signal = SIZE_HALF
            elif rsi_7_oversold and bull_trend_1w:
                new_signal = SIZE_HALF
        
        # Path 4: KAMA trend + RSI pullback
        if bull_trend_1w and kama_bullish:
            if rsi_neutral and close[i] > ema_21[i]:
                new_signal = SIZE_BASE
        
        # Path 5: Simple momentum (loose condition)
        if bull_trend_1w and hma_bullish:
            if di_bullish and close[i] > ema_21[i]:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Trend following in trending regime (primary)
        if trending_regime and bear_trend_1w:
            if hma_bearish and di_bearish:
                if strong_trend:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
        
        # Path 2: EMA crossover + trend bias
        if bear_trend_1w and ema_bearish:
            if rsi[i] > 30 and rsi[i] < 60:
                new_signal = -SIZE_BASE
        
        # Path 3: Mean reversion in ranging regime
        if ranging_regime:
            if rsi_overbought and near_bb_upper:
                new_signal = -SIZE_HALF
            elif rsi_7_overbought and bear_trend_1w:
                new_signal = -SIZE_HALF
        
        # Path 4: KAMA trend + RSI pullback
        if bear_trend_1w and kama_bearish:
            if rsi_neutral and close[i] < ema_21[i]:
                new_signal = -SIZE_BASE
        
        # Path 5: Simple momentum (loose condition)
        if bear_trend_1w and hma_bearish:
            if di_bearish and close[i] < ema_21[i]:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals