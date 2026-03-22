#!/usr/bin/env python3
"""
Experiment #008: 30m Fisher Transform + KAMA Adaptive Trend + 4h HMA Filter
Hypothesis: 30m timeframe captures intraday swings with less noise than 15m.
Fisher Transform (Ehlers) excels at identifying reversals in ranging markets.
KAMA (Kaufman Adaptive MA) adapts to volatility - smooth in ranges, responsive in trends.
Choppiness Index filters regime: CHOP>61.8 = range (use Fisher reversals), CHOP<38.2 = trend (use KAMA breakouts).
4h HMA provides HTF trend bias to avoid counter-trend trades that fail in strong trends.
Key innovation: Combines 3 different signal types (Fisher, KAMA, CHOP) with regime switching.
Position sizing: 0.25 base, 0.35 max for strong signals, discrete levels to minimize fee churn.
Stoploss: 2.5*ATR trailing stop to limit drawdown during crashes.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_kama_4h_hma_regime_v1"
timeframe = "30m"
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
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er[:er_period] = np.nan
    
    # Smoothing constant
    sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """Calculate Ehlers Fisher Transform for reversal detection."""
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    hl2 = (high + low) / 2.0
    
    # Highest high and lowest low over period
    hh = np.zeros(n)
    ll = np.zeros(n)
    for i in range(period - 1, n):
        hh[i] = np.max(hl2[i - period + 1:i + 1])
        ll[i] = np.min(hl2[i - period + 1:i + 1])
    
    # Normalize price
    norm = np.zeros(n)
    mask = (hh - ll) > 0
    norm[mask] = 0.33 * 2.0 * (hl2[mask] - ll[mask]) / (hh[mask] - ll[mask]) - 1.0
    norm = np.clip(norm, -0.99, 0.99)
    
    # Fisher transform
    fisher_raw = 0.5 * np.log((1 + norm) / (1 - norm))
    
    # Smooth with EMA
    fisher_s = pd.Series(fisher_raw).ewm(span=3, min_periods=3, adjust=False).mean().values
    fisher[period:] = fisher_s[period:]
    
    return fisher

def calculate_choppiness_index(high, low, close, period=14):
    """Calculate Choppiness Index for regime detection."""
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        
        if (hh - ll) > 0 and sum_tr > 0:
            chop[i] = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(period)
    
    return chop

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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean().values
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher = calculate_fisher_transform(high, low, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.35
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # Regime detection via Choppiness Index
        # CHOP > 61.8 = ranging (mean reversion), CHOP < 38.2 = trending
        ranging_regime = chop[i] > 55
        trending_regime = chop[i] < 45
        neutral_regime = 45 <= chop[i] <= 55
        
        # Fisher Transform signals (reversal detection)
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > -1.5 and fisher[i-1] <= -1.5 if not np.isnan(fisher[i-1]) else False
        fisher_cross_down = fisher[i] < 1.5 and fisher[i-1] >= 1.5 if not np.isnan(fisher[i-1]) else False
        
        # KAMA trend signals
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        kama_cross_up = close[i] > kama[i] and close[i-1] <= kama[i-1] if not np.isnan(kama[i-1]) else False
        kama_cross_down = close[i] < kama[i] and close[i-1] >= kama[i-1] if not np.isnan(kama[i-1]) else False
        
        # MACD momentum
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        macd_cross_up = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if not np.isnan(macd_hist[i-1]) else False
        macd_cross_down = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if not np.isnan(macd_hist[i-1]) else False
        
        # RSI extremes for mean reversion
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and close[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and close[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55): Fisher Transform Mean Reversion ===
        if ranging_regime:
            # Long: Fisher oversold + HTF bull trend or neutral
            if fisher_oversold and (bull_trend or not bear_trend):
                new_signal = SIZE_BASE
            # Short: Fisher overbought + HTF bear trend or neutral
            elif fisher_overbought and (bear_trend or not bull_trend):
                new_signal = -SIZE_BASE
            # Stronger: Fisher cross with RSI confirmation
            elif fisher_cross_up and rsi_oversold:
                new_signal = SIZE_BASE
            elif fisher_cross_down and rsi_overbought:
                new_signal = -SIZE_BASE
        
        # === TRENDING REGIME (CHOP < 45): KAMA/MACD Trend Following ===
        elif trending_regime:
            # Long: KAMA cross up + MACD bullish + HTF bull trend
            if kama_cross_up and macd_bullish and bull_trend:
                new_signal = SIZE_MAX
            # Short: KAMA cross down + MACD bearish + HTF bear trend
            elif kama_cross_down and macd_bearish and bear_trend:
                new_signal = -SIZE_MAX
            # Weaker: Just KAMA cross with HTF trend
            elif kama_cross_up and bull_trend:
                new_signal = SIZE_BASE
            elif kama_cross_down and bear_trend:
                new_signal = -SIZE_BASE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55): Conservative ===
        elif neutral_regime:
            # Only take strongest signals with multiple confirmations
            if kama_bullish and macd_bullish and bull_trend and rsi_oversold:
                new_signal = SIZE_BASE
            elif kama_bearish and macd_bearish and bear_trend and rsi_overbought:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals