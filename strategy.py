#!/usr/bin/env python3
"""
Experiment #030: 1d Multi-Timeframe Trend-Follow with 1w HMA Regime Filter
Hypothesis: Daily timeframe captures major trends while weekly HMA provides robust regime filter.
Key insight: 1d has fewer whipsaws than lower TFs. Weekly HMA smoother than daily for regime detection.
Entry logic: 1d EMA pullback entries aligned with 1w trend bias, RSI confirmation for timing.
Position sizing: 0.25-0.30 discrete levels to minimize fee churn while controlling drawdown.
Timeframe: 1d (REQUIRED for exp#030), HTF: 1w via mtf_data helper.
Why this might work: 1d captures major moves, 1w HMA filters out counter-trend trades in strong regimes.
Must generate 10+ trades on train, 3+ on test - entry conditions loosened vs failed experiments.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_trend_1w_hma_ema_rsi_v1"
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

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
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
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    zscore = calculate_zscore(close, 20)
    
    # Donchian channels for breakout detection
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # HMA on 1d for faster trend
    hma_1d = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
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
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 1w trend bias (HTF) - main regime filter
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # 1d trend confirmation
        bull_trend_1d = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_1d = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # RSI conditions - LOOSENED for more trades on 1d
        rsi_pullback_long = 35 < rsi[i] < 65
        rsi_bounce_short = 35 < rsi[i] < 65
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        # Z-score filter - avoid extreme entries
        zscore_neutral = abs(zscore[i]) < 2.5
        
        # HMA crossover on 1d
        hma_cross_long = False
        hma_cross_short = False
        if i >= 1 and not np.isnan(hma_1d[i]) and not np.isnan(hma_1d[i-1]):
            hma_cross_long = hma_1d[i] > ema_50[i] and hma_1d[i-1] <= ema_50[i-1]
            hma_cross_short = hma_1d[i] < ema_50[i] and hma_1d[i-1] >= ema_50[i-1]
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.03 and close[i] >= ema_21[i] * 0.97
        price_near_ema21_short = close[i] >= ema_21[i] * 0.97 and close[i] <= ema_21[i] * 1.03
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 3:
            higher_low = low[i] > low[i-3]
            lower_high = high[i] < high[i-3]
        
        # Donchian breakout signals
        donch_breakout_long = close[i] > donch_upper[i-1] if not np.isnan(donch_upper[i-1]) else False
        donch_breakout_short = close[i] < donch_lower[i-1] if not np.isnan(donch_lower[i-1]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 1w bullish) ===
        if bull_trend_1w:
            # Primary: Pullback to EMA21 with RSI confirmation
            if price_near_ema21_long and rsi_pullback_long and above_200:
                new_signal = SIZE_BASE
            
            # Secondary: HMA crossover with 1w confirmation
            elif hma_cross_long and bull_trend_1w and bull_trend_1d:
                new_signal = SIZE_BASE
            
            # Tertiary: RSI oversold bounce in uptrend
            elif rsi_oversold and bull_trend_1d and zscore_neutral:
                new_signal = SIZE_HALF
            
            # Momentum: Higher low with trend
            elif higher_low and bull_trend_1d and rsi[i] > 40:
                new_signal = SIZE_HALF
            
            # Donchian breakout with trend
            elif donch_breakout_long and bull_trend_1w and above_200:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRIES (only when 1w bearish) ===
        elif bear_trend_1w:
            # Primary: Bounce to EMA21 with RSI confirmation
            if price_near_ema21_short and rsi_bounce_short and below_200:
                new_signal = -SIZE_BASE
            
            # Secondary: HMA crossover with 1w confirmation
            elif hma_cross_short and bear_trend_1w and bear_trend_1d:
                new_signal = -SIZE_BASE
            
            # Tertiary: RSI overbought rejection in downtrend
            elif rsi_overbought and bear_trend_1d and zscore_neutral:
                new_signal = -SIZE_HALF
            
            # Momentum: Lower high with trend
            elif lower_high and bear_trend_1d and rsi[i] < 60:
                new_signal = -SIZE_HALF
            
            # Donchian breakdown with trend
            elif donch_breakout_short and bear_trend_1w and below_200:
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