#!/usr/bin/env python3
"""
Experiment #018: 1d Regime-Adaptive Strategy with Bollinger Width + ADX Filter
Hypothesis: Previous trend-following strategies failed because they didn't adapt to market regime.
This strategy uses Bollinger Band Width percentile to detect range vs trend, then applies:
- Range regime (BB Width < 30th percentile): Mean reversion at BB bounds with RSI confirmation
- Trend regime (BB Width > 70th percentile + ADX > 25): Trend follow on pullbacks
Key insight: 2022 was whipsaw (range→trend→range), 2025 is bear/range. One strategy doesn't fit all.
Timeframe: 1d (REQUIRED for exp#018), single timeframe with regime detection.
Position sizing: 0.30 for strong signals, 0.15 for weaker, discrete levels to minimize churn.
Stoploss: 2.5*ATR trailing stop to protect against 2022-style crashes.
Why this might work: Adapts to market state instead of forcing one approach. Generates trades in both regimes.
"""
import numpy as np
import pandas as pd

name = "regime_adaptive_1d_bb_adx_rsi_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = upper - lower
    pct_b = (close - lower) / (width + 1e-10)
    return upper, lower, sma, width, pct_b

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
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
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm / (atr + 1e-10)).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = 100 * pd.Series(minus_dm / (atr + 1e-10)).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_percentile_rank(values, period=100):
    """Calculate rolling percentile rank."""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = values[i-period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            pr[i] = np.sum(valid < values[i]) / len(valid)
    
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Calculate all indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_sma, bb_width, pct_b = calculate_bollinger(close, 20, 2.0)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Bollinger Width percentile for regime detection
    bb_width_pct = calculate_percentile_rank(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_STRONG = 0.30
    SIZE_WEAK = 0.15
    
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
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Range regime: BB Width in lower 30th percentile (compression)
        is_range_regime = bb_width_pct[i] < 0.30
        
        # Trend regime: BB Width in upper 30th percentile + ADX > 20
        is_trend_regime = bb_width_pct[i] > 0.70 and adx[i] > 20
        
        # Neutral regime: everything else
        is_neutral = not is_range_regime and not is_trend_regime
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # EMA trend
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === RANGE REGIME: MEAN REVERSION ===
        if is_range_regime:
            # Long: Price at lower BB + RSI oversold
            if pct_b[i] < 0.15 and rsi[i] < 35:
                new_signal = SIZE_STRONG
            # Long: Price near lower BB + RSI rising from oversold
            elif pct_b[i] < 0.30 and rsi[i] < 40 and rsi[i] > rsi[i-1]:
                new_signal = SIZE_WEAK
            # Short: Price at upper BB + RSI overbought
            elif pct_b[i] > 0.85 and rsi[i] > 65:
                new_signal = -SIZE_STRONG
            # Short: Price near upper BB + RSI falling from overbought
            elif pct_b[i] > 0.70 and rsi[i] > 60 and rsi[i] < rsi[i-1]:
                new_signal = -SIZE_WEAK
            else:
                new_signal = 0.0
        
        # === TREND REGIME: TREND FOLLOWING ===
        elif is_trend_regime:
            # Long: Pullback to EMA21 in uptrend + RSI bounce
            if ema_bullish and above_200:
                if close[i] <= ema_21[i] * 1.02 and close[i] >= ema_21[i] * 0.97:
                    if rsi[i] > 40 and rsi[i] < 60 and rsi[i] > rsi[i-1]:
                        new_signal = SIZE_STRONG
                # Momentum continuation
                elif close[i] > ema_21[i] and plus_di[i] > minus_di[i] and adx[i] > 25:
                    if rsi[i] > 45 and rsi[i] < 65:
                        new_signal = SIZE_WEAK
                else:
                    new_signal = 0.0
            # Short: Bounce to EMA21 in downtrend + RSI rejection
            elif ema_bearish and below_200:
                if close[i] >= ema_21[i] * 0.97 and close[i] <= ema_21[i] * 1.02:
                    if rsi[i] > 40 and rsi[i] < 60 and rsi[i] < rsi[i-1]:
                        new_signal = -SIZE_STRONG
                # Momentum continuation
                elif close[i] < ema_21[i] and minus_di[i] > plus_di[i] and adx[i] > 25:
                    if rsi[i] > 35 and rsi[i] < 55:
                        new_signal = -SIZE_WEAK
                else:
                    new_signal = 0.0
            else:
                new_signal = 0.0
        
        # === NEUTRAL REGIME: REDUCED POSITION SIZING ===
        else:
            # Only take strongest signals in neutral
            if pct_b[i] < 0.10 and rsi[i] < 30:
                new_signal = SIZE_WEAK
            elif pct_b[i] > 0.90 and rsi[i] > 70:
                new_signal = -SIZE_WEAK
            else:
                new_signal = 0.0
        
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