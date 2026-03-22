#!/usr/bin/env python3
"""
Experiment #194: 30m RSI Mean Reversion + 4h HMA Trend Filter + BB Regime + ATR Stop

Hypothesis: 30m timeframe captures short-term mean reversion opportunities while
4h HMA provides stable higher-timeframe trend bias. RSI extremes (35/65) with
Bollinger Band confirmation create high-probability entries. This combines
trend-following (HTF bias) with mean-reversion (RSI extremes) which has shown
promise in past experiments.

Why 30m might work:
- 30m = 48 bars/day, enough frequency for mean reversion without 5m noise
- RSI mean reversion works better on shorter timeframes than trend-following
- 4h HMA filter prevents counter-trend trades (major failure mode in past exp)
- BB width regime filter avoids entering during squeezes (low vol = fakeouts)
- Relaxed RSI thresholds (35/65 vs 30/70) ensure sufficient trade count

Learning from failures:
- #182, #192: Donchian breakouts failed (too many false breakouts)
- #183, #190: Vol spike strategies failed (0 trades or negative Sharpe)
- #187: Supertrend failed (whipsaws in ranges)
- #190: 0 trades = auto reject (must ensure entry conditions not too strict)
- Mean reversion works on 30m/1h, trend-following works on 4h/12h/1d

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_bb_4h_hma_meanrev_atr_v1"
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
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands (middle, upper, lower)."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return middle.values, upper.values, lower.values, std.values

def calculate_bb_width(upper, lower, middle):
    """Calculate Bollinger Band Width (normalized)."""
    bb_width = (upper - lower) / (middle + 1e-10)
    return bb_width

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
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_percentile_rank(series, window=100):
    """Calculate Percentile Rank for Connors RSI component."""
    n = len(series)
    pr = np.zeros(n)
    for i in range(window, n):
        window_data = series[i-window+1:i+1]
        current = series[i]
        pr[i] = np.sum(window_data < current) / window * 100
    return pr

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
    rsi = calculate_rsi(close, 14)
    bb_mid, bb_upper, bb_lower, bb_std = calculate_bollinger_bands(close, 20, 2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_mid)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Calculate BB Width percentile for regime detection
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
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
        
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_width_pr[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (BB Width Percentile) ===
        # BB Width < 30th percentile = squeeze (avoid entries)
        # BB Width > 30th percentile = normal/expansion (allow entries)
        regime_ok = bb_width_pr[i] > 30
        
        # === RSI MEAN REVERSION ===
        # Long: RSI < 35 (oversold) in uptrend
        # Short: RSI > 65 (overbought) in downtrend
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # === BOLLINGER BAND CONFIRMATION ===
        # Long: price near or below lower band
        # Short: price near or above upper band
        bb_long_confirm = close[i] <= bb_lower[i] * 1.005  # Within 0.5% of lower
        bb_short_confirm = close[i] >= bb_upper[i] * 0.995  # Within 0.5% of upper
        
        # === EMA STRUCTURE ===
        # Long: EMA21 > EMA50 (bullish structure)
        # Short: EMA21 < EMA50 (bearish structure)
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + regime ok + RSI oversold + (BB confirm OR EMA bullish)
        # Relaxed conditions to ensure enough trades
        if bull_trend_4h and regime_ok and rsi_oversold:
            # Need at least one confirmation (BB or EMA structure)
            if bb_long_confirm or ema_bullish:
                new_signal = SIZE_BASE
        
        # Short: 4h bearish + regime ok + RSI overbought + (BB confirm OR EMA bearish)
        if bear_trend_4h and regime_ok and rsi_overbought:
            # Need at least one confirmation (BB or EMA structure)
            if bb_short_confirm or ema_bearish:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals