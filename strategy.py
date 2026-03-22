#!/usr/bin/env python3
"""
Experiment #012: 1d Multi-Timeframe Trend-Follow with 1w HMA Regime Filter
Hypothesis: Daily timeframe captures major trends while weekly HMA provides ultra-long-term regime bias.
Key insight: Previous 1d strategy (exp#006 Donchian) had negative Sharpe. This uses EMA pullback entries + RSI timing + BB width filter.
Why this might work: 1d has fewer whipsaws, 1w HMA smoother than 1d for regime, BB width avoids low-vol traps.
Position sizing: 0.25-0.30 discrete levels, ATR 2.5x stoploss to control drawdown.
Timeframe: 1d (REQUIRED for exp#012), HTF: 1w via mtf_data helper.
Entry logic loosened to ensure 10+ trades on train, 3+ on test.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_ema_rsi_1w_hma_bb_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / (sma + 1e-10)
    return upper, lower, width

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram."""
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    change = np.abs(close - np.roll(close, er_period))
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-er_period):i+1])))
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    
    sc = (er * (2.0/(fast_period+1) - 2.0/(slow_period+1)) + 2.0/(slow_period+1)) ** 2
    
    kama[er_period] = close[er_period]
    for i in range(er_period+1, n):
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
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, 20, 2.0)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    kama = calculate_kama(close, 10, 2, 30)
    
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
    
    # BB width percentile for regime (volatility filter)
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        # 1w trend bias (HTF) - ultra-long-term regime filter
        bull_regime_1w = close[i] > hma_1w_aligned[i]
        bear_regime_1w = close[i] < hma_1w_aligned[i]
        
        # 1d trend confirmation
        bull_trend_1d = ema_21[i] > ema_50[i] and close[i] > ema_50[i]
        bear_trend_1d = ema_21[i] < ema_50[i] and close[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(ema_200[i]) and close[i] > ema_200[i]
        below_200 = not np.isnan(ema_200[i]) and close[i] < ema_200[i]
        
        # KAMA adaptive trend
        kama_bull = not np.isnan(kama[i]) and close[i] > kama[i]
        kama_bear = not np.isnan(kama[i]) and close[i] < kama[i]
        
        # RSI conditions - LOOSENED for more trades on daily
        rsi_pullback_long = 35 < rsi[i] < 65
        rsi_bounce_short = 35 < rsi[i] < 65
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        # MACD momentum confirmation
        macd_bull = macd_hist[i] > 0
        macd_bear = macd_hist[i] < 0
        
        # Bollinger Band width filter (avoid low-vol traps)
        vol_ok = not np.isnan(bb_width_ma[i]) and bb_width[i] > bb_width_ma[i] * 0.7
        
        # Price position within BB
        price_near_lower = close[i] < bb_lower[i] * 1.02
        price_near_upper = close[i] > bb_upper[i] * 0.98
        
        # EMA pullback entry
        price_near_ema21_long = close[i] <= ema_21[i] * 1.03 and close[i] >= ema_21[i] * 0.97
        price_near_ema21_short = close[i] >= ema_21[i] * 0.97 and close[i] <= ema_21[i] * 1.03
        
        # EMA crossover signals
        ema_cross_long = False
        ema_cross_short = False
        if i >= 1 and not np.isnan(ema_21[i]) and not np.isnan(ema_21[i-1]):
            ema_cross_long = ema_21[i] > ema_50[i] and ema_21[i-1] <= ema_50[i-1]
            ema_cross_short = ema_21[i] < ema_50[i] and ema_21[i-1] >= ema_50[i-1]
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 3:
            higher_low = low[i] > low[i-3]
            lower_high = high[i] < high[i-3]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 1w bullish regime) ===
        if bull_regime_1w:
            # Primary: EMA pullback with RSI confirmation
            if price_near_ema21_long and rsi_pullback_long and above_200:
                new_signal = SIZE_BASE
            
            # Secondary: EMA crossover with momentum
            elif ema_cross_long and macd_bull and vol_ok:
                new_signal = SIZE_BASE
            
            # Tertiary: RSI oversold bounce in uptrend
            elif rsi_oversold and bull_trend_1d and kama_bull:
                new_signal = SIZE_HALF
            
            # Momentum: Higher low with trend
            elif higher_low and bull_trend_1d and rsi[i] > 40:
                new_signal = SIZE_HALF
            
            # BB bounce: Price near lower band in uptrend
            elif price_near_lower and bull_trend_1d and rsi[i] < 50:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 1w bearish regime) ===
        elif bear_regime_1w:
            # Primary: EMA bounce with RSI confirmation
            if price_near_ema21_short and rsi_bounce_short and below_200:
                new_signal = -SIZE_BASE
            
            # Secondary: EMA crossover with momentum
            elif ema_cross_short and macd_bear and vol_ok:
                new_signal = -SIZE_BASE
            
            # Tertiary: RSI overbought rejection in downtrend
            elif rsi_overbought and bear_trend_1d and kama_bear:
                new_signal = -SIZE_HALF
            
            # Momentum: Lower high with trend
            elif lower_high and bear_trend_1d and rsi[i] < 60:
                new_signal = -SIZE_HALF
            
            # BB rejection: Price near upper band in downtrend
            elif price_near_upper and bear_trend_1d and rsi[i] > 50:
                new_signal = -SIZE_HALF
        
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