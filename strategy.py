#!/usr/bin/env python3
"""
Experiment #104: 30m KAMA Adaptive Trend + 4h HMA Filter + RSI Pullback + Volume
Hypothesis: 30m timeframe needs adaptive trend following (KAMA) to handle crypto volatility.
4h HMA provides stable trend bias without whipsaws. RSI pullback entries (not extremes)
ensure we enter on dips in established trends. Volume confirmation filters false breakouts.
ATR stoploss at 2.0x protects capital during reversals.

Why this might work on 30m (learning from failures):
- #092 (30m CRSI mean reversion): Sharpe=-3.147 — mean reversion fails on 30m
- #098 (30m EMA crossover): Sharpe=-0.836 — simple EMA too slow, too many whipsaws
- #100 (4h KAMA adaptive): Sharpe=0.436 — KAMA works! Adapt to volatility
- Key insight: KAMA adapts to market noise, slows in chop, speeds in trends
- 4h HMA filter prevents counter-trend trades on 30m
- Volume ratio > 1.2 confirms real moves vs fakeouts
- Conservative sizing (0.20-0.30) limits drawdown

Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
Position sizing: 0.20 base, 0.30 strong signals. Stoploss at 2.0*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_4h_hma_rsi_pullback_volume_v1"
timeframe = "30m"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise: fast in trends, slow in chop.
    ER (Efficiency Ratio) = |price change| / sum of individual changes
    SC (Smoothing Constant) = ER * (fast_SC - slow_SC) + slow_SC
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    price_change = np.abs(close - np.roll(close, er_period))
    price_change[:er_period] = np.nan
    
    sum_changes = np.zeros(n)
    for i in range(er_period, n):
        sum_changes[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = np.zeros(n)
    mask = sum_changes > 0
    er[mask] = price_change[mask] / sum_changes[mask]
    er[:er_period] = np.nan
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate SC (Smoothing Constant)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc[:er_period] = np.nan
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    dx = np.zeros(n)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio[:period] = np.nan
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias (stable, avoids whipsaws)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === KAMA ADAPTIVE TREND ===
        # KAMA adapts to volatility: fast in trends, slow in chop
        kama_bullish = close[i] > kama[i] and kama[i] > kama[i-1] if i > 0 else False
        kama_bearish = close[i] < kama[i] and kama[i] < kama[i-1] if i > 0 else False
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ADX REGIME FILTER (avoid choppy markets) ===
        # ADX > 18 = trending market (lower for 30m to ensure trades)
        trending_market = adx[i] > 18
        strong_trend = adx[i] > 25
        
        # === RSI PULLBACK (not extremes - catch entries in trend) ===
        # For longs: RSI 35-55 (pullback in uptrend)
        # For shorts: RSI 45-65 (pullback in downtrend)
        rsi_pullback_long = 35 <= rsi[i] <= 55
        rsi_pullback_short = 45 <= rsi[i] <= 65
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 40
        rsi_momentum_short = rsi[i] < 60
        
        # === VOLUME CONFIRMATION ===
        # Volume ratio > 1.2 confirms real moves
        volume_confirmed = vol_ratio[i] > 1.0
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Path 1: Strong trend + 4h bullish + KAMA bullish + volume (primary)
        if strong_trend and bull_trend_4h and kama_bullish:
            if volume_confirmed and (ema_bullish or rsi_momentum_long):
                new_signal = SIZE_STRONG
            elif ema_bullish or rsi_pullback_long:
                new_signal = SIZE_BASE
        
        # Path 2: 4h bullish + KAMA bullish + trending (simpler, ensures trades)
        if new_signal == 0.0 and bull_trend_4h and kama_bullish and trending_market:
            if ema_bullish or rsi_pullback_long:
                new_signal = SIZE_BASE
        
        # Path 3: 4h bullish + EMA bullish + RSI momentum (fallback)
        if new_signal == 0.0 and bull_trend_4h and ema_bullish and rsi_momentum_long:
            if trending_market:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Path 1: Strong trend + 4h bearish + KAMA bearish + volume (primary)
        if strong_trend and bear_trend_4h and kama_bearish:
            if volume_confirmed and (ema_bearish or rsi_momentum_short):
                new_signal = -SIZE_STRONG
            elif ema_bearish or rsi_pullback_short:
                new_signal = -SIZE_BASE
        
        # Path 2: 4h bearish + KAMA bearish + trending (simpler, ensures trades)
        if new_signal == 0.0 and bear_trend_4h and kama_bearish and trending_market:
            if ema_bearish or rsi_pullback_short:
                new_signal = -SIZE_BASE
        
        # Path 3: 4h bearish + EMA bearish + RSI momentum (fallback)
        if new_signal == 0.0 and bear_trend_4h and ema_bearish and rsi_momentum_short:
            if trending_market:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR for 30m ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.0 * ATR below highest close
            stoploss_price = highest_close - 2.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.0 * ATR above lowest close
            stoploss_price = lowest_close + 2.0 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals