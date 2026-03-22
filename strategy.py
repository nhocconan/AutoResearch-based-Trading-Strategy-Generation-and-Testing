#!/usr/bin/env python3
"""
Experiment #144: 1d HMA Trend + 1w HTF Bias + RSI Momentum + ATR Stop

Hypothesis: Daily timeframe provides cleaner signals with less noise than intraday.
Using 1w HTF for broad trend bias, HMA(21) for primary trend, RSI(14) for momentum
timing, and ATR(14) trailing stop for risk management.

Why this might work on 1d:
- Daily bars filter out intraday noise and fake breakouts
- 1w HTF provides macro trend context (critical for 2022 crash avoidance)
- RSI momentum ensures we enter on pullbacks, not chased tops
- Simple logic = fewer conditions that could block all trades (learned from #142, #143)
- ATR stop adapts to volatility regime automatically

Key improvements over failed strategies:
- LESS filtering than #142/#143 (those got 0 trades)
- HMA smoother than EMA for trend detection
- 1w HTF bias prevents counter-trend trades in major crashes
- Position sizing 0.20-0.30 discrete levels limits drawdown

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_1w_rsi_momentum_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    if n < period * 2:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = tr_s > 0
    plus_di[mask] = 100 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100 * minus_dm_s[mask] / tr_s[mask]
    
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

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
    hma_1d = calculate_hma(close, 21)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = higher timeframe trend bias (macro trend)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === 1d HMA TREND ===
        # Price above HMA = bullish momentum
        bull_hma_1d = close[i] > hma_1d[i]
        bear_hma_1d = close[i] < hma_1d[i]
        
        # HMA slope (momentum confirmation)
        hma_slope = hma_1d[i] - hma_1d[i-5] if i >= 5 else 0
        hma_bull_slope = hma_slope > 0
        hma_bear_slope = hma_slope < 0
        
        # === RSI MOMENTUM ===
        # RSI > 50 = bullish momentum, RSI < 50 = bearish
        rsi_bull = rsi[i] > 50
        rsi_bear = rsi[i] < 50
        
        # RSI extremes for stronger signals
        rsi_strong_bull = rsi[i] > 55
        rsi_strong_bear = rsi[i] < 45
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 20  # Trending market
        adx_weak = adx[i] <= 20   # Ranging market
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: 1w bullish + 1d HMA bullish + HMA slope up + RSI strong bull + ADX strong
        if bull_trend_1w and bull_hma_1d and hma_bull_slope and rsi_strong_bull and adx_strong:
            new_signal = SIZE_STRONG
        # Moderate: 1w bullish + 1d HMA bullish + RSI bull
        elif bull_trend_1w and bull_hma_1d and rsi_bull:
            new_signal = SIZE_BASE
        # Weak (ensure trades): 1w bullish + 1d HMA bullish (minimal filter)
        elif bull_trend_1w and bull_hma_1d:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: 1w bearish + 1d HMA bearish + HMA slope down + RSI strong bear + ADX strong
        if bear_trend_1w and bear_hma_1d and hma_bear_slope and rsi_strong_bear and adx_strong:
            new_signal = -SIZE_STRONG
        # Moderate: 1w bearish + 1d HMA bearish + RSI bear
        elif bear_trend_1w and bear_hma_1d and rsi_bear:
            new_signal = -SIZE_BASE
        # Weak (ensure trades): 1w bearish + 1d HMA bearish (minimal filter)
        elif bear_trend_1w and bear_hma_1d:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals