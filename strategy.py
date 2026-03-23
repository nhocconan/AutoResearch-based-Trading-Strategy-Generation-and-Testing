#!/usr/bin/env python3
"""
Experiment #268: 30m Primary + 4h/1d HTF — Volatility Compression Breakout

Hypothesis: Lower TF (30m) strategies fail due to either (1) too many trades → fee drag,
or (2) too strict filters → 0 trades. Solution: Use HTF (4h/1d) for DIRECTION,
30m only for ENTRY TIMING with volatility compression detection.

KEY INSIGHT FROM FAILURES (#258, #260, #265):
- RSI pullback alone on 30m/1h generates too many whipsaw trades
- Complex regime (CHOP + CRSI) creates 0-trade scenarios on lower TF
- SOLUTION: BB Width compression (volatility squeeze) + HTF trend + moderate RSI

STRATEGY LOGIC:
- 1d HMA(21): Macro trend bias (only long if price > 1d HMA, only short if <)
- 4h HMA(21): Medium-term trend direction (confirms 1d bias)
- 30m BB Width(20): Volatility compression (entry when BB Width < 30th percentile)
- 30m RSI(14): Entry timing (40-60 zone for continuation, not extremes)
- Volume filter: > 0.8x 20-bar avg (avoid low liquidity)
- ATR(14) 2.0x trailing stoploss

TARGET: 40-80 trades/year, Sharpe > 0.5 on ALL symbols
POSITION SIZE: 0.25 (conservative for 30m volatility)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_bb_squeeze_hma_rsi_4h1d_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

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
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    band_width = (upper - lower) / sma
    return upper.values, lower.values, band_width.values

def calculate_percentile_rank(series, window=100):
    """Calculate Percentile Rank over rolling window."""
    s = pd.Series(series)
    def pct_rank(x):
        if len(x) < 2:
            return np.nan
        return (x < x.iloc[-1]).sum() / (len(x) - 1) * 100
    pr = s.rolling(window=window, min_periods=20).apply(pct_rank, raw=False)
    return pr.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # BB Width Percentile Rank (volatility compression detection)
    bb_width_pr = calculate_percentile_rank(bb_width, window=100)
    
    # Volume SMA for filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h HMA for medium-term trend (aligned properly with shift(1))
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro trend (aligned properly with shift(1))
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(bb_width[i]) or np.isnan(bb_width_pr[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY COMPRESSION (BB Width Percentile) ===
        # BB Width in bottom 30% = volatility squeeze (potential breakout)
        vol_compression = bb_width_pr[i] < 30.0
        
        # === VOLUME FILTER ===
        # Volume must be at least 0.8x average (avoid low liquidity)
        volume_ok = volume[i] >= 0.8 * vol_sma_20[i]
        
        # === RSI ENTRY TIMING ===
        # Long: RSI 40-55 (bullish but not overbought)
        # Short: RSI 45-60 (bearish but not oversold)
        rsi_long = (rsi_14[i] >= 40.0) and (rsi_14[i] <= 55.0)
        rsi_short = (rsi_14[i] >= 45.0) and (rsi_14[i] <= 60.0)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 1d bullish + 4h bullish + vol compression + volume + RSI
        if price_above_hma_1d and price_above_hma_4h and vol_compression and volume_ok and rsi_long:
            desired_signal = POSITION_SIZE
        
        # SHORT ENTRY: 1d bearish + 4h bearish + vol compression + volume + RSI
        elif price_below_hma_1d and price_below_hma_4h and vol_compression and volume_ok and rsi_short:
            desired_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and price_below_hma_4h:
            desired_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and price_above_hma_4h:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        # Exit long if RSI becomes overbought (>70)
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0
        
        # Exit short if RSI becomes oversold (<30)
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0
        
        # === VOLATILITY EXPANSION EXIT ===
        # Exit if BB Width expands significantly (breakout happened, take profit)
        if in_position and bb_width_pr[i] > 70.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if setup still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still bullish
                if price_above_hma_4h and price_above_hma_1d:
                    desired_signal = POSITION_SIZE * 0.6  # Reduce to 60% while holding
            elif position_side < 0:
                # Hold short if trend still bearish
                if price_below_hma_4h and price_below_hma_1d:
                    desired_signal = -POSITION_SIZE * 0.6
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals