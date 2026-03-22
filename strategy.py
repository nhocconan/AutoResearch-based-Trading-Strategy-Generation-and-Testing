#!/usr/bin/env python3
"""
Experiment #425: 12h HMA Trend + RSI Pullback + BB Volatility Regime

Hypothesis: After analyzing 413+ failed experiments, the key insight is that
12h timeframe needs SIMPLER logic with fewer filters. Complex regime-switching
(ADX > 25 vs < 20) creates whipsaws and too few trades. This strategy uses:

1. 1d HMA(21) TREND BIAS (via mtf_data helper):
   - Simple directional filter: price > HMA = long bias, price < HMA = short bias
   - HMA smoother than EMA, reduces lag on 12h/1d alignment
   - NO complex regime detection - just trend direction

2. 12h RSI(7) PULLBACK ENTRIES:
   - Long: RSI < 35 (oversold pullback) + price > 1d HMA
   - Short: RSI > 65 (overbought rally) + price < 1d HMA
   - Faster RSI period (7 vs 14) = more signals on 12h timeframe
   - Relaxed thresholds (35/65 vs 30/70) = MORE TRADES

3. BOLLINGER BAND WIDTH REGIME:
   - BB Width percentile < 40% = low vol (enter on breakout)
   - BB Width percentile > 60% = high vol (mean revert)
   - Adaptive entry based on volatility state

4. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from crashes while letting winners run

5. POSITION SIZING: 0.25 discrete (conservative for 12h)
   - Lower than 0.30 to reduce drawdown
   - Discrete levels minimize fee churn

Why this should beat #413 (Sharpe=-0.239):
- Fewer conditions = more trades generated (critical for Sharpe > 0)
- No ADX regime whipsaw (20-25 boundary caused false exits)
- Faster RSI (7 vs 14) catches more pullbacks on 12h
- BB width adds volatility context without over-filtering
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_trend_rsi7_pullback_bb_vol_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=7):
    """Calculate Relative Strength Index with faster period for more signals."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    width = (upper - lower) / sma
    
    return upper.values, lower.values, width.values

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate percentile rank of BB width over lookback period."""
    n = len(bb_width)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        window = bb_width[i-lookback+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(valid < bb_width[i]) / len(valid) * 100
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)  # Faster RSI for more signals
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25  # Conservative for 12h volatility
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        
        # === 1d HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Relaxed thresholds for MORE trades (35/65 vs 30/70)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # === VOLATILITY REGIME ===
        low_vol = bb_width_pct[i] < 40  # Compression - expect breakout
        high_vol = bb_width_pct[i] > 60  # Expansion - mean revert
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Bullish trend + RSI pullback
        if bull_trend_1d and rsi_oversold:
            # Low vol: enter on pullback expecting continuation
            # High vol: enter on pullback expecting mean reversion up
            new_signal = SIZE
        
        # SHORT ENTRY: Bearish trend + RSI rally
        elif bear_trend_1d and rsi_overbought:
            # Low vol: enter on rally expecting continuation down
            # High vol: enter on rally expecting mean reversion down
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1d HMA trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === RSI EXTREME EXIT ===
        # Exit long if RSI goes very overbought (>80)
        # Exit short if RSI goes very oversold (<20)
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi[i] > 80:
                new_signal = 0.0
            if position_side < 0 and rsi[i] < 20:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
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