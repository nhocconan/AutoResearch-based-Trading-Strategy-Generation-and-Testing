#!/usr/bin/env python3
"""
Experiment #257: 12h Fisher Transform Mean Reversion with 1d/1w HMA Bias

Hypothesis: 12h timeframe is ideal for swing mean-reversion trades (3-7 day holds).
Fisher Transform excels at catching reversals at extremes, especially in bear/range markets.
Combined with Bollinger Band extremes and 1d/1w HMA for soft directional bias.

Why this might work:
- 12h captures multi-day swings without 1d's slowness
- Fisher Transform normalizes price to Gaussian distribution, extremes = reversal points
- Bollinger Bands confirm volatility extremes
- 1d/1w HMA provides soft bias (doesn't block entries, just adjusts conviction)
- Mean reversion works better than trend-following in 2022-2024 bear/range markets
- LOOSE entry thresholds to ensure trades happen (learned from #250, #244 failures)

Key differences from failed experiments:
- #245 (12h KAMA): Sharpe=-0.580 - trend following in bear market
- #251 (12h ADX RSI): Sharpe=-0.342 - too many filters
- This uses Fisher Transform (proven reversal indicator) + BB extremes
- Fewer conflicting filters = more trades
- Mean reversion focus (works in bear markets where trend following fails)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_bb_meanrev_1d_1w_hma_atr_v1"
timeframe = "12h"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Extreme values (>1.5 or <-1.5) indicate reversal points.
    """
    hl2 = (high + low) / 2
    hl2_s = pd.Series(hl2)
    
    # Calculate highest high and lowest low over period
    n = len(hl2)
    highest = np.zeros(n)
    lowest = np.zeros(n)
    
    for i in range(period, n):
        highest[i] = np.max(hl2[i-period+1:i+1])
        lowest[i] = np.min(hl2[i-period+1:i+1])
    
    highest[:period] = np.nan
    lowest[:period] = np.nan
    
    # Normalize to 0-1 range
    epsilon = 1e-10
    norm = (hl2 - lowest) / (highest - lowest + epsilon)
    norm = np.clip(norm, 0.001, 0.999)  # Avoid log(0) or log(1)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + norm) / (1 - norm + epsilon))
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    
    # Smooth with EMA
    fisher_s = pd.Series(fisher)
    fisher_smooth = fisher_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return fisher_smooth

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher_transform(high, low, 9)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    rsi_7 = calculate_rsi(close, 7)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(bb_upper[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long = fisher[i] > -1.5 and fisher[i-1] <= -1.5
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short = fisher[i] < 1.5 and fisher[i-1] >= 1.5
        
        # === BOLLINGER BAND EXTREMES ===
        # Long: Price at or below lower band (oversold)
        bb_long = close[i] <= bb_lower[i]
        # Short: Price at or above upper band (overbought)
        bb_short = close[i] >= bb_upper[i]
        
        # === RSI CONFIRMATION ===
        # Long: RSI < 35 (oversold)
        rsi_long = rsi_7[i] < 35
        # Short: RSI > 65 (overbought)
        rsi_short = rsi_7[i] > 65
        
        # === ENTRY SIGNALS (LOOSE CONDITIONS TO ENSURE TRADES) ===
        new_signal = 0.0
        
        # Long entry: Need Fisher reversal OR BB extreme + RSI confirmation
        long_score = 0
        if fisher_long:
            long_score += 1.5  # Fisher is primary signal
        if bb_long:
            long_score += 1.0
        if rsi_long:
            long_score += 0.5
        
        # Short entry
        short_score = 0
        if fisher_short:
            short_score += 1.5
        if bb_short:
            short_score += 1.0
        if rsi_short:
            short_score += 0.5
        
        # Entry threshold: score >= 1.5 (allows Fisher alone or BB+RSI)
        if long_score >= 1.5:
            new_signal = SIZE_BASE
            # Boost if 1d trend agrees
            if bull_trend_1d:
                new_signal = min(new_signal + 0.05, 0.30)
        
        if short_score >= 1.5:
            new_signal = -SIZE_BASE
            # Boost if 1d trend agrees
            if bear_trend_1d:
                new_signal = max(new_signal - 0.05, -0.30)
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * atr[entry_price_idx]:
                    new_signal = SIZE_HALF
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * atr[entry_price_idx]:
                    new_signal = -SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals