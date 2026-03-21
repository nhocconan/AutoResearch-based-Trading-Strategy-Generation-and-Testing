#!/usr/bin/env python3
"""
Experiment #009: 1h Fisher Transform + BB Regime + 4h Trend Filter
Hypothesis: Fisher Transform catches reversals in bear/range markets (2025 test period).
Bollinger Band Width detects regime (squeeze = breakout, wide = trend).
4h HMA provides major trend filter to avoid counter-trend trades.
This differs from previous HMA/RSI/Supertrend combos by using Fisher for entries.
Position sizing: 0.25 discrete with 2.5*ATR stoploss to control drawdown.
Relaxed entry thresholds to ensure ≥10 trades/symbol on train data.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_bb_4h_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals when Fisher crosses extreme levels (-1.5/+1.5).
    Works well in bear/range markets for mean reversion entries.
    """
    hl2 = (high + low) / 2
    # Normalize price to 0-1 range
    highest = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    range_val = highest - lowest
    range_val = np.where(range_val == 0, 1e-6, range_val)  # avoid div by zero
    normalized = (hl2 - lowest) / range_val
    normalized = np.clip(normalized, 0.001, 0.999)  # avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width for regime detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    # Band Width = (Upper - Lower) / SMA (normalized)
    bw = (upper - lower) / sma
    bw = np.nan_to_num(bw, nan=1.0)
    # BB Percentile - where price sits within bands
    bb_pct = (close - lower) / (upper - lower)
    bb_pct = np.where((upper - lower) == 0, 0.5, bb_pct)
    bb_pct = np.clip(bb_pct, 0, 1)
    bb_pct = np.nan_to_num(bb_pct, nan=0.5)
    return upper, lower, bw, bb_pct

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    bb_upper, bb_lower, bb_width, bb_pct = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    
    # BB Width percentile for regime detection (rolling 100 periods)
    bb_width_pct = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: np.percentile(x, 50), raw=True
    ).values
    bb_width_pct = np.nan_to_num(bb_width_pct, nan=50.0)
    
    # Volume confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=1.0)
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    max_close = np.zeros(n)
    min_close = np.zeros(n)
    
    for i in range(100, n):
        # 4h trend filter (major regime)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Regime detection from BB Width
        # Low BW (<30th percentile) = squeeze = breakout coming
        # High BW (>70th percentile) = trend = mean revert
        regime_squeeze = bb_width_pct[i] < 35
        regime_trend = bb_width_pct[i] > 65
        
        # Fisher Transform signals
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # RSI confirmation (relaxed for more trades)
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        # BB position confirmation
        bb_low = bb_pct[i] < 0.35  # price in lower band
        bb_high = bb_pct[i] > 0.65  # price in upper band
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # Entry logic - multiple paths to ensure trades
        new_signal = 0.0
        
        # Path 1: Fisher reversal with 4h trend support (primary)
        if trend_4h_bullish and fisher_long and rsi_oversold:
            new_signal = SIZE
        elif trend_4h_bearish and fisher_short and rsi_overbought:
            new_signal = -SIZE
        
        # Path 2: BB squeeze breakout (regime-based)
        elif regime_squeeze and bb_low and rsi[i] < 40 and vol_confirm:
            new_signal = SIZE
        elif regime_squeeze and bb_high and rsi[i] > 60 and vol_confirm:
            new_signal = -SIZE
        
        # Path 3: Mean reversion in trending regime
        elif regime_trend and bb_low and fisher[i] < -1.0:
            new_signal = SIZE
        elif regime_trend and bb_high and fisher[i] > 1.0:
            new_signal = -SIZE
        
        # Path 4: Simple Fisher extreme (fallback for more trades)
        elif fisher[i] < -2.0 and fisher_signal[i] < fisher[i]:
            new_signal = SIZE
        elif fisher[i] > 2.0 and fisher_signal[i] > fisher[i]:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Update max close for trailing
                max_close[i] = max(max_close[i-1] if i > 0 else 0, close[i])
                trailing_stop[i] = max_close[i] - 2.5 * atr[i]
                if close[i] < trailing_stop[i] and trailing_stop[i] > 0:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] > entry_price[i-1] + 2.5 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Update min close for trailing
                min_close[i] = min(min_close[i-1] if i > 0 else 999999, close[i])
                trailing_stop[i] = min_close[i] + 2.5 * atr[i]
                if close[i] > trailing_stop[i] and trailing_stop[i] < 999999:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] < entry_price[i-1] - 2.5 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            max_close[i] = close[i] if position_side > 0 else max_close[i-1] if i > 0 else 0
            min_close[i] = close[i] if position_side < 0 else min_close[i-1] if i > 0 else 999999
            trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                max_close[i] = close[i] if position_side > 0 else max_close[i-1] if i > 0 else 0
                min_close[i] = close[i] if position_side < 0 else min_close[i-1] if i > 0 else 999999
                trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            else:
                max_close[i] = max_close[i-1] if i > 0 else max_close[i]
                min_close[i] = min_close[i-1] if i > 0 else min_close[i]
                trailing_stop[i] = trailing_stop[i-1] if i > 0 else trailing_stop[i]
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            max_close[i] = max_close[i-1] if i > 0 else 0
            min_close[i] = min_close[i-1] if i > 0 else 999999
            trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals