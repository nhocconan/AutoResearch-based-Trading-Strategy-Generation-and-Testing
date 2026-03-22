#!/usr/bin/env python3
"""
Experiment #001: 15m Regime-Adaptive Multi-Timeframe Strategy
Hypothesis: 15m timeframe captures intraday moves while 4h/1h HTF filters prevent counter-trend trades.
Key innovation: Bollinger Band Width detects regime (range vs trend), switching between mean-reversion
and trend-following logic. 4h HMA provides primary bias, 1h BB Width detects choppiness, 15m RSI+Stoch
for precise entries. Multiple entry paths (6 long + 6 short) ensure >=10 trades per symbol.
Conservative sizing (0.25-0.30) with 2.5*ATR stoploss for 15m volatility. Must beat Sharpe=0.499 baseline.
Timeframe: 15m (REQUIRED), HTF: 4h and 1h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_adaptive_bb_rsi_stoch_v1"
timeframe = "15m"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bw = (upper - lower) / sma * 100  # Band Width as percentage
    pct_b = (close - lower) / (upper - lower + 1e-10)  # Position within bands
    return upper, lower, sma, bw, pct_b

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator."""
    n = len(close)
    stoch_k = np.zeros(n)
    stoch_k[:] = np.nan
    
    for i in range(k_period, n):
        lowest_low = np.min(low[i-k_period+1:i+1])
        highest_high = np.max(high[i-k_period+1:i+1])
        if highest_high > lowest_low:
            stoch_k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
        else:
            stoch_k[i] = 50.0
    
    stoch_d = pd.Series(stoch_k).rolling(window=d_period, min_periods=d_period).mean().values
    return stoch_k, stoch_d

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        change = np.abs(close[i] - close[i-period])
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if volatility > 0:
            er[i] = change / volatility
    
    # Smoothing Constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    kama[period] = close[period]
    for i in range(period+1, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    bb_1h_upper, bb_1h_lower, bb_1h_sma, bb_1h_bw, bb_1h_pct = calculate_bollinger_bands(df_1h['close'].values, 20, 2.0)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    bb_1h_bw_aligned = align_htf_to_ltf(prices, df_1h, bb_1h_bw)
    bb_1h_pct_aligned = align_htf_to_ltf(prices, df_1h, bb_1h_pct)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_bw, bb_pct = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    stoch_k, stoch_d = calculate_stochastic(high, low, close, 14, 3)
    adx = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    kama = calculate_kama(close, 10, 2, 30)
    
    # Calculate BB Width percentile for regime detection (rolling 100 bars)
    bb_bw_percentile = pd.Series(bb_bw).rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x < x[-1]) / len(x) * 100 if len(x) > 0 else 50
    ).values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(bb_1h_bw_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(stoch_k[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_pct[i]) or np.isnan(bb_bw[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h Regime detection via BB Width
        # BW > 61.8 percentile = trending, BW < 38.2 percentile = ranging
        regime_trending = bb_1h_bw_aligned[i] > 4.0  # Absolute threshold for 1h
        regime_ranging = bb_1h_bw_aligned[i] < 2.5
        
        # 1h RSI bias
        rsi_1h_bullish = rsi_1h_aligned[i] > 50
        rsi_1h_bearish = rsi_1h_aligned[i] < 50
        
        # 15m BB position
        bb_near_lower = bb_pct[i] < 0.15
        bb_near_upper = bb_pct[i] > 0.85
        bb_squeeze = bb_bw[i] < bb_bw_percentile[i] * 0.01 if not np.isnan(bb_bw_percentile[i]) else False
        
        # 15m RSI zones
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 50
        rsi_pullback_short = rsi[i] > 50 and rsi[i] < 65
        
        # 15m Stochastic
        stoch_oversold = stoch_k[i] < 20 and stoch_d[i] < 20
        stoch_overbought = stoch_k[i] > 80 and stoch_d[i] > 80
        stoch_cross_up = stoch_k[i] > stoch_d[i] and stoch_k[i-1] <= stoch_d[i-1] if i > 0 else False
        stoch_cross_down = stoch_k[i] < stoch_d[i] and stoch_k[i-1] >= stoch_d[i-1] if i > 0 else False
        
        # EMA trend
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        trend_weak = adx[i] < 20
        
        # KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        new_signal = 0.0
        
        # === TRENDING REGIME ENTRIES (follow the trend) ===
        
        # Long Path 1: 4h HMA bullish + 1h trending + 15m RSI pullback + Stoch cross up
        if regime_trending and hma_4h_bullish and rsi_1h_bullish and rsi_pullback_long and stoch_cross_up:
            new_signal = SIZE_ENTRY
        
        # Long Path 2: 4h HMA bullish + EMA bullish + ADX strong + Stoch cross up
        elif hma_4h_bullish and ema_bullish and trend_strong and stoch_cross_up and adx[i] > adx[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Long Path 3: 4h HMA bullish + KAMA bullish + RSI > 45 + Stoch > 20
        elif hma_4h_bullish and kama_bullish and rsi[i] > 45 and stoch_k[i] > 20 and stoch_d[i] > 20:
            new_signal = SIZE_ENTRY
        
        # Short Path 1: 4h HMA bearish + 1h trending + 15m RSI pullback + Stoch cross down
        if regime_trending and hma_4h_bearish and rsi_1h_bearish and rsi_pullback_short and stoch_cross_down:
            new_signal = -SIZE_ENTRY
        
        # Short Path 2: 4h HMA bearish + EMA bearish + ADX strong + Stoch cross down
        elif hma_4h_bearish and ema_bearish and trend_strong and stoch_cross_down and adx[i] > adx[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Short Path 3: 4h HMA bearish + KAMA bearish + RSI < 55 + Stoch < 80
        elif hma_4h_bearish and kama_bearish and rsi[i] < 55 and stoch_k[i] < 80 and stoch_d[i] < 80:
            new_signal = -SIZE_ENTRY
        
        # === RANGING REGIME ENTRIES (mean reversion) ===
        
        # Long Path 4: Range regime + BB near lower + RSI oversold + Stoch oversold
        if regime_ranging and bb_near_lower and rsi_oversold and stoch_oversold:
            new_signal = SIZE_ENTRY * 0.8  # Smaller size for mean reversion
        
        # Long Path 5: Range regime + BB near lower + Stoch cross up + 4h not bearish
        elif regime_ranging and bb_near_lower and stoch_cross_up and not hma_4h_bearish:
            new_signal = SIZE_ENTRY * 0.8
        
        # Short Path 4: Range regime + BB near upper + RSI overbought + Stoch overbought
        if regime_ranging and bb_near_upper and rsi_overbought and stoch_overbought:
            new_signal = -SIZE_ENTRY * 0.8
        
        # Short Path 5: Range regime + BB near upper + Stoch cross down + 4h not bullish
        elif regime_ranging and bb_near_upper and stoch_cross_down and not hma_4h_bullish:
            new_signal = -SIZE_ENTRY * 0.8
        
        # === BREAKOUT ENTRIES (BB squeeze expansion) ===
        
        # Long Path 6: BB squeeze + price breaks above BB upper + 4h bullish
        if bb_squeeze and close[i] > bb_upper[i] and hma_4h_bullish and volume_confirmed(prices, i):
            new_signal = SIZE_ENTRY
        
        # Short Path 6: BB squeeze + price breaks below BB lower + 4h bearish
        elif bb_squeeze and close[i] < bb_lower[i] and hma_4h_bearish and volume_confirmed(prices, i):
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 15m timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 15m timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals

def volume_confirmed(prices, i):
    """Check if volume confirms the breakout (above 20-bar average)."""
    if i < 20:
        return True
    volume = prices['volume'].values
    avg_vol = np.mean(volume[i-20:i])
    return volume[i] > avg_vol * 1.2