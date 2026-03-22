#!/usr/bin/env python3
"""
Experiment #006: Daily Trend-Following with Weekly Bias
Hypothesis: Daily timeframe captures major crypto trends while weekly HMA provides 
macro trend bias to avoid counter-trend trades. Simpler than regime-adaptive approaches
that failed in exp#002-005. Key: relaxed entry filters to ensure sufficient trades
(need 10+ in train, 3+ in test) while maintaining trend alignment.

Strategy: Weekly HMA(21) for bull/bear market bias + Daily HMA(13/34) crossover 
for entries + RSI(14) momentum filter. ATR(14) trailing stop at 2.5x for risk control.
Position sizing: 0.25 base, 0.35 max with HTF confirmation.

Why this should work on 1d:
- Daily bars filter intraday noise that killed 15m/30m strategies
- Weekly trend bias prevents entering against major trend (2022 crash protection)
- Faster HMA (13/34 vs 21/50) generates more signals for minimum trade requirement
- RSI filter relaxed (45/55 not 30/70) to avoid missing trends

Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_1w_trend_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    # Use 1w for major trend bias on daily strategy
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate daily indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Daily HMA for faster crossover signals
    hma_13 = calculate_hma(close, 13)
    hma_34 = calculate_hma(close, 34)
    hma_50 = calculate_hma(close, 50)
    
    # EMA for additional confirmation
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_13[i]) or np.isnan(hma_34[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (HTF) - major market direction
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # Daily HMA crossover signals (faster than 21/50)
        hma_cross_long = hma_13[i] > hma_34[i] and hma_13[i - 1] <= hma_34[i - 1]
        hma_cross_short = hma_13[i] < hma_34[i] and hma_13[i - 1] >= hma_34[i - 1]
        
        # Daily HMA position (already crossed)
        hma_bullish = hma_13[i] > hma_34[i]
        hma_bearish = hma_13[i] < hma_34[i]
        
        # RSI momentum filter (relaxed for more trades)
        rsi_bullish = rsi[i] > 45
        rsi_bearish = rsi[i] < 55
        rsi_strong_bull = rsi[i] > 50
        rsi_strong_bear = rsi[i] < 50
        
        # EMA confirmation
        ema_bullish = close[i] > ema_21[i] and close[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and close[i] < ema_50[i]
        
        # Price above/below HMA50 for trend confirmation
        price_above_hma50 = close[i] > hma_50[i]
        price_below_hma50 = close[i] < hma_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Strong long: Weekly bull + Daily HMA cross + RSI bull + EMA bull
        if hma_cross_long and bull_trend_1w and rsi_bullish and ema_bullish:
            new_signal = SIZE_MAX
        # Moderate long: Weekly bull + Daily HMA bull + RSI bull
        elif hma_bullish and bull_trend_1w and rsi_bullish and price_above_hma50:
            new_signal = SIZE_BASE
        # Weak long: Weekly bull + HMA cross (RSI filter relaxed)
        elif hma_cross_long and bull_trend_1w:
            new_signal = SIZE_BASE
        # Trend continuation: Weekly bull + HMA already bull + RSI rising
        elif bull_trend_1w and hma_bullish and rsi[i] > rsi[i - 3] and i >= 3:
            if signals[i - 1] > 0:  # Already long, maintain
                new_signal = SIZE_BASE
        
        # === SHORT ENTRIES ===
        # Strong short: Weekly bear + Daily HMA cross + RSI bear + EMA bear
        if hma_cross_short and bear_trend_1w and rsi_bearish and ema_bearish:
            new_signal = -SIZE_MAX
        # Moderate short: Weekly bear + Daily HMA bear + RSI bear
        elif hma_bearish and bear_trend_1w and rsi_bearish and price_below_hma50:
            new_signal = -SIZE_BASE
        # Weak short: Weekly bear + HMA cross
        elif hma_cross_short and bear_trend_1w:
            new_signal = -SIZE_BASE
        # Trend continuation: Weekly bear + HMA already bear + RSI falling
        elif bear_trend_1w and hma_bearish and rsi[i] < rsi[i - 3] and i >= 3:
            if signals[i - 1] < 0:  # Already short, maintain
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position trailing stop
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position trailing stop
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
            # New position entry
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            # Position reversal
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            # Position exit
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals