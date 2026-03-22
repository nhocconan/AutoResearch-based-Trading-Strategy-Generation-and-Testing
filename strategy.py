#!/usr/bin/env python3
"""
Experiment #064: 4h Dual-Regime Strategy with 1d/1w HMA Trend Filter
Hypothesis: 4h timeframe offers good balance for regime-switching strategies.
Key insight: Use ADX to switch between trend-following (ADX>25) and mean-reversion (ADX<20).
1d HMA provides primary trend bias, 1w HMA provides macro bias for additional filter.
KAMA for adaptive trend following (less whipsaw than EMA in ranging markets).
Bollinger Band mean reversion for ranging regime entries.
Ehlers Fisher Transform for precise reversal timing in mean-reversion mode.
Why this might work: Adapts to market regime instead of using one approach always.
Entry conditions loosened to ensure 10+ trades per symbol on train, 3+ on test.
Position sizing: 0.25 base, 0.35 strong signal, discrete levels to minimize fees.
Timeframe: 4h (REQUIRED), HTF: 1d and 1w via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_1d_1w_hma_kama_fisher_v1"
timeframe = "4h"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    signal = np.abs(close - np.roll(close, er_period))
    signal[:er_period] = np.nan
    noise = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=er_period, min_periods=er_period).sum().values
    noise[:er_period] = np.nan
    
    er = np.zeros(n)
    er[:] = np.nan
    mask = noise > 0
    er[mask] = signal[mask] / noise[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform for reversal detection.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X is normalized price.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    # Typical price
    typical = (high + low + close) / 3
    
    # Normalize price over lookback period
    for i in range(period, n):
        window = typical[i-period+1:i+1]
        highest = np.max(window)
        lowest = np.min(window)
        
        if highest != lowest:
            x = 2 * (typical[i] - lowest) / (highest - lowest) - 1
            x = np.clip(x, -0.999, 0.999)  # Prevent ln domain error
            fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
    
    # Signal line (1-period lag)
    fisher_signal[period+1:] = fisher[period:-1]
    
    return fisher, fisher_signal

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / (sma + 1e-10) * 100
    return upper, lower, bandwidth, sma

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # KAMA for adaptive trend
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=10)
    kama_slow = calculate_kama(close, er_period=10, fast_period=5, slow_period=30)
    
    # Bollinger Bands for mean reversion
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    
    # Fisher Transform for reversals
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    
    # EMA for trend confirmation
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = intermediate trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA = macro trend bias
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # Combined trend strength (both HTF agree = stronger signal)
        strong_bull = bull_trend_1d and bull_trend_1w
        strong_bear = bear_trend_1d and bear_trend_1w
        
        # === REGIME DETECTION ===
        trending_regime = adx[i] > 25
        strong_trending = adx[i] > 30
        ranging_regime = adx[i] < 20
        
        # DI crossover for trend direction
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === KAMA TREND SIGNALS ===
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # EMA alignment
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # Price vs SMA200
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] <= bb_lower[i] * 1.005 if not np.isnan(bb_lower[i]) else False
        near_bb_upper = close[i] >= bb_upper[i] * 0.995 if not np.isnan(bb_upper[i]) else False
        bb_squeeze = bb_bandwidth[i] < np.nanpercentile(bb_bandwidth[:i], 30) if i > 100 else False
        
        # === FISHER TRANSFORM REVERSAL ===
        fisher_long_signal = False
        fisher_short_signal = False
        if not np.isnan(fisher[i]) and not np.isnan(fisher_signal[i]):
            # Fisher crossing above -1.5 from below
            fisher_long_signal = fisher_signal[i] < -1.5 and fisher[i] > -1.5
            # Fisher crossing below +1.5 from above
            fisher_short_signal = fisher_signal[i] > 1.5 and fisher[i] < 1.5
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_extreme_oversold = rsi[i] < 25
        rsi_extreme_overbought = rsi[i] > 75
        
        new_signal = 0.0
        
        # === TRENDING REGIME (ADX > 25) - Follow trend ===
        if trending_regime:
            # Long: KAMA bullish + HTF bull + DI bullish
            if bull_trend_1d and kama_bullish and di_bullish:
                if strong_bull:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
            
            # Short: KAMA bearish + HTF bear + DI bearish
            if bear_trend_1d and kama_bearish and di_bearish:
                if strong_bear:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
        
        # === RANGING REGIME (ADX < 20) - Mean reversion ===
        if ranging_regime:
            # Long: Near BB lower + RSI oversold + Fisher long signal + above SMA200
            if near_bb_lower and rsi_oversold:
                if fisher_long_signal or rsi_extreme_oversold:
                    if above_sma200 or bull_trend_1d:
                        new_signal = SIZE_HALF
            
            # Short: Near BB upper + RSI overbought + Fisher short signal + below SMA200
            if near_bb_upper and rsi_overbought:
                if fisher_short_signal or rsi_extreme_overbought:
                    if below_sma200 or bear_trend_1d:
                        new_signal = -SIZE_HALF
        
        # === TRANSITION REGIME (ADX 20-25) - Use RSI pullback ===
        if not trending_regime and not ranging_regime:
            # Long pullback in uptrend
            if bull_trend_1d and ema_bullish:
                if rsi[i] > 40 and rsi[i] < 55:
                    if close[i] > ema_21[i]:
                        new_signal = SIZE_BASE
            
            # Short pullback in downtrend
            if bear_trend_1d and ema_bearish:
                if rsi[i] > 45 and rsi[i] < 60:
                    if close[i] < ema_21[i]:
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