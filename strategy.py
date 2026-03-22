#!/usr/bin/env python3
"""
Experiment #154: 4h Bollinger Mean Reversion + 1d HMA Trend Filter + ATR Stop

Hypothesis: In bear/range markets (2022 crash, 2025 bear), mean reversion at 
Bollinger Band extremes outperforms trend following. 1d HMA provides trend bias 
to avoid counter-trend mean reversion (which destroys capital in strong trends).

Why this might work:
- BTC 2025+ is bear/range (-25%), trend strategies failing (exp #145-153 all negative)
- Bollinger mean reversion has 65-70% win rate in range markets
- 4h timeframe balances trade frequency with signal quality
- 1d HMA filter prevents dangerous counter-trend entries
- Simple conditions ensure adequate trade count (≥10 per symbol)

Learning from failures:
- #145-153: All negative Sharpe - pure trend strategies failing in current regime
- Need mean-reversion entries with trend filter for bear/range markets
- Too many filters = 0 trades (#142, #143), keep conditions simple

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_rsi_1d_hma_meanrev_atr_v1"
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands with proper min_periods."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === BOLLINGER + RSI MEAN REVERSION SIGNAL ===
        # Long: Price at lower BB + RSI oversold + 1d trend bullish (or neutral)
        # Short: Price at upper BB + RSI overbought + 1d trend bearish (or neutral)
        
        price_at_lower_bb = close[i] <= bb_lower[i] * 1.005  # Within 0.5% of lower band
        price_at_upper_bb = close[i] >= bb_upper[i] * 0.995  # Within 0.5% of upper band
        
        rsi_oversold = rsi[i] < 35  # Loose threshold for more trades
        rsi_overbought = rsi[i] > 65  # Loose threshold for more trades
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Mean reversion long: at lower BB + oversold RSI + 1d not strongly bearish
        if price_at_lower_bb and rsi_oversold:
            if bull_trend_1d:
                new_signal = SIZE_STRONG  # Strong signal with trend
            else:
                new_signal = SIZE_BASE  # Weaker signal against trend
        
        # === SHORT ENTRY CONDITIONS ===
        # Mean reversion short: at upper BB + overbought RSI + 1d not strongly bullish
        if price_at_upper_bb and rsi_overbought:
            if bear_trend_1d:
                new_signal = -SIZE_STRONG  # Strong signal with trend
            else:
                new_signal = -SIZE_BASE  # Weaker signal against trend
        
        # === EXIT CONDITIONS ===
        # Exit long when price reaches middle BB or RSI neutral
        if in_position and position_side > 0:
            if close[i] >= bb_mid[i] or rsi[i] > 55:
                new_signal = 0.0
        
        # Exit short when price reaches middle BB or RSI neutral
        if in_position and position_side < 0:
            if close[i] <= bb_mid[i] or rsi[i] < 45:
                new_signal = 0.0
        
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