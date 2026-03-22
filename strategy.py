#!/usr/bin/env python3
"""
Experiment #250: 4h Sentiment Reversal Strategy with Taker Volume + RSI + 1d HMA

Hypothesis: 4h timeframe captures multi-day sentiment extremes that reverse.
Using taker_buy_volume ratio as sentiment proxy (crowd positioning) +
RSI for entry timing + 1d HMA for trend bias.

Why this might work on 4h:
- Taker buy ratio > 0.65 = crowd overly long (reversal likely)
- Taker buy ratio < 0.35 = crowd overly short (bounce likely)
- 4h captures sentiment extremes without 15m/1h noise
- 1d HMA provides trend bias but doesn't block counter-trend reversals
- Simpler entry conditions = more trades (critical lesson from failures)
- Conservative sizing (0.25) + ATR stoploss controls drawdown

Key improvements over failed experiments:
- #244 (4h Fisher): 0 trades - conditions too strict
- #238 (4h Chop/Connors): Sharpe=-0.056 - too many filters
- This uses LOOSE thresholds: RSI 35/65 (not 20/80), taker ratio 0.35/0.65
- Only 3 conditions for entry (not 5-7 conflicting filters)
- Allows counter-trend trades when sentiment extreme (not just trend-follow)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_sentiment_rsi_1d_hma_atr_v1"
timeframe = "4h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_taker_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (sentiment proxy)."""
    ratio = np.zeros(len(volume))
    mask = volume > 0
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    taker_ratio = calculate_taker_ratio(taker_buy_volume, volume)
    
    # Calculate 4h HMA for local trend
    hma_4h = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.12
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_price_idx = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(taker_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = trend bias (but we allow counter-trend on extreme sentiment)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === SENTIMENT EXTREMES ===
        # Taker ratio > 0.65 = crowd overly long (look for short)
        # Taker ratio < 0.35 = crowd overly short (look for long)
        sentiment_long_extreme = taker_ratio[i] > 0.65
        sentiment_short_extreme = taker_ratio[i] < 0.35
        
        # === RSI CONFIRMATION ===
        # RSI > 65 = overbought (supports short)
        # RSI < 35 = oversold (supports long)
        rsi_overbought = rsi_14[i] > 65
        rsi_oversold = rsi_14[i] < 35
        
        # === LOCAL TREND ===
        hma_4h_bullish = close[i] > hma_4h[i]
        hma_4h_bearish = close[i] < hma_4h[i]
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Sentiment short extreme + RSI oversold ---
        # Can enter even against 1d trend if sentiment is extreme enough
        if sentiment_short_extreme and rsi_oversold:
            # Stronger signal if 1d trend is bullish or neutral
            if bull_trend_1d or (not bear_trend_1d):
                new_signal = SIZE_BASE
            # Still enter counter-trend if sentiment VERY extreme
            elif taker_ratio[i] < 0.30 and rsi_14[i] < 30:
                new_signal = SIZE_BASE
        
        # --- SHORT ENTRY: Sentiment long extreme + RSI overbought ---
        if sentiment_long_extreme and rsi_overbought:
            # Stronger signal if 1d trend is bearish or neutral
            if bear_trend_1d or (not bull_trend_1d):
                new_signal = -SIZE_BASE
            # Still enter counter-trend if sentiment VERY extreme
            elif taker_ratio[i] > 0.70 and rsi_14[i] > 70:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
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
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * atr[entry_price_idx]:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * atr[entry_price_idx]:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly reduced size)
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