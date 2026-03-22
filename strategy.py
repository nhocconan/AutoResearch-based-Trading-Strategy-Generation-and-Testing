#!/usr/bin/env python3
"""
Experiment #105: 1h KAMA Trend + 4h HMA Filter + Supertrend Entry + ADX Regime
Hypothesis: 1h timeframe needs adaptive trend following (KAMA) with HTF bias (4h HMA).
KAMA adapts to volatility (fast in trends, slow in ranges) - better than EMA for crypto.
4h HMA provides stable trend bias without excessive lag.
Supertrend(10,3) gives precise entry timing on pullbacks.
ADX regime filter avoids choppy markets (ADX>18 for 1h, lower than daily).

Why this might work on 1h (learning from failures #093, #097, #099):
- Fisher transform failed twice on 1h (-0.452, -1.835 Sharpe) - avoid it
- RSI mean reversion on 1h was deadly (-6.373 Sharpe) - don't mean revert on 1h
- Simple EMA crossovers always fail on BTC/ETH - use KAMA instead
- Need simpler entry conditions to ensure trades (learned from #094 with 0 trades)
- 1h is Goldilocks zone: faster than 4h, less noise than 15m/30m

Key improvements:
- KAMA instead of EMA (adaptive to volatility, proven in #100 with 0.436 Sharpe)
- 4h HMA for trend bias (faster than 1d, more signals than 1d)
- Lower ADX threshold (18 vs 25) to ensure trades on all symbols
- Discrete position sizing (0.0, ±0.25, ±0.35) to minimize fee churn
- Trailing stoploss at 2.0*ATR (tighter for 1h vs 2.5*ATR for 1d)

Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.35 strong signals. Stoploss at 2.0*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_supertrend_adx_regime_v1"
timeframe = "1h"
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
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - fast in trends, slow in ranges.
    Proven to work well in crypto (see #100 with 0.436 Sharpe).
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er[:er_period] = np.nan
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = er ** 2 * (fast_sc - slow_sc) + slow_sc
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
    
    return adx, plus_di, minus_di

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if np.isnan(atr[i]):
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
            
        if direction[i-1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            if close[i] < supertrend[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            if close[i] > supertrend[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # KAMA - adaptive trend following (proven in #100)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Supertrend for entry timing
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # Simple EMA for additional confirmation
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
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
        
        if np.isnan(adx[i]) or np.isnan(kama[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === KAMA TREND (adaptive) ===
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        kama_slope_up = kama[i] > kama[i-5] if i >= 5 else False
        kama_slope_down = kama[i] < kama[i-5] if i >= 5 else False
        
        # === SUPERTREND SIGNAL ===
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ADX REGIME FILTER (avoid choppy markets) ===
        # ADX > 18 = trending market (lower threshold for 1h to ensure trades)
        # ADX > 25 = strong trending market
        trending_market = adx[i] > 18
        strong_trend = adx[i] > 25
        
        # === RSI MOMENTUM (not extremes - ensure trades) ===
        rsi_bullish = rsi[i] > 45
        rsi_bearish = rsi[i] < 55
        rsi_neutral = 35 <= rsi[i] <= 65
        
        # === DI CONFIRMATION ===
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (simplified to ensure trades) ===
        # Path 1: Strong trend - all indicators align (SIZE_STRONG)
        if st_bullish and bull_trend_4h and kama_bullish and strong_trend:
            if ema_bullish and rsi_bullish:
                new_signal = SIZE_STRONG
            elif ema_bullish or rsi_bullish:
                new_signal = SIZE_BASE
        
        # Path 2: Moderate trend - most indicators align (SIZE_BASE)
        if new_signal == 0.0 and st_bullish and bull_trend_4h and trending_market:
            if kama_bullish and (ema_bullish or rsi_bullish):
                new_signal = SIZE_BASE
        
        # Path 3: Simple trend - KAMA + Supertrend + 4h bias (fallback for trades)
        if new_signal == 0.0 and st_bullish and kama_bullish and bull_trend_4h:
            if trending_market or di_bullish:
                new_signal = SIZE_BASE
        
        # Path 4: Minimal conditions - ensure we get trades on all symbols
        if new_signal == 0.0 and st_bullish and bull_trend_4h:
            if kama_slope_up and rsi_neutral:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (simplified to ensure trades) ===
        # Path 1: Strong trend - all indicators align (SIZE_STRONG)
        if st_bearish and bear_trend_4h and kama_bearish and strong_trend:
            if ema_bearish and rsi_bearish:
                new_signal = -SIZE_STRONG
            elif ema_bearish or rsi_bearish:
                new_signal = -SIZE_BASE
        
        # Path 2: Moderate trend - most indicators align (SIZE_BASE)
        if new_signal == 0.0 and st_bearish and bear_trend_4h and trending_market:
            if kama_bearish and (ema_bearish or rsi_bearish):
                new_signal = -SIZE_BASE
        
        # Path 3: Simple trend - KAMA + Supertrend + 4h bias (fallback for trades)
        if new_signal == 0.0 and st_bearish and kama_bearish and bear_trend_4h:
            if trending_market or di_bearish:
                new_signal = -SIZE_BASE
        
        # Path 4: Minimal conditions - ensure we get trades on all symbols
        if new_signal == 0.0 and st_bearish and bear_trend_4h:
            if kama_slope_down and rsi_neutral:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR for 1h ===
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