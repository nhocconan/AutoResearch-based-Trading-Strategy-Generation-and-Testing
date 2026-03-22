#!/usr/bin/env python3
"""
Experiment #014: 30m MACD Momentum + 4h HMA Trend + BB Volatility + Volume

Hypothesis: After analyzing 13 failed experiments, the pattern shows:
1. Mean reversion (CRSI, RSI extremes) fails in crypto's persistent trending nature
2. ONLY #002 (Supertrend + 4h HMA + RSI) achieved positive Sharpe=0.123
3. Complex regime filters (Choppiness) added noise, not alpha
4. Funding data alignment caused issues on multiple symbols

This 30m strategy uses a DIFFERENT combination than #002:

1. 4h HMA trend bias: Proven in #002. Only long if price>4h_HMA, only short if price<4h_HMA.
   Call get_htf_data() ONCE before loop, use align_htf_to_ltf() for proper alignment.

2. MACD Histogram momentum: Different from RSI. Long when MACD hist crosses above 0,
   short when crosses below 0. Captures momentum shifts with less whipsaw than RSI.

3. Bollinger Band Width volatility: BB Width > median = high vol (trend), BB Width < median = low vol (breakout setup).
   Enter on breakout from low vol compression.

4. Volume confirmation: Volume > 1.5 * 20-period MA volume = confirmed move.
   Filters false breakouts.

5. ATR trailing stop: 2.5 * ATR(14) to protect from reversals.

Why this should beat #002 (Sharpe=0.123):
- MACD histogram smoother than RSI for momentum (less whipsaw)
- BB Width identifies compression before breakout (better entry timing)
- Volume filter reduces false signals (critical on 30m)
- Same proven 4h HMA trend filter as #002
- Target 40-60 trades/year on 30m (optimal per Rule 10)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_macd_4h_hma_bb_vol_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_macd(close, fast=12, slow=26, signal=9):
    """
    Calculate MACD line, Signal line, and Histogram.
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal
    """
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """
    Calculate Bollinger Bands.
    Middle = SMA(period)
    Upper = Middle + std_dev * StdDev(period)
    Lower = Middle - std_dev * StdDev(period)
    Width = (Upper - Lower) / Middle
    """
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle
    return upper.values, lower.values, width.values

def calculate_volume_ma(volume, period=20):
    """Calculate moving average of volume."""
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=period, min_periods=period).mean()
    return vol_ma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    # Calculate BB Width median for regime detection
    bb_width_median = np.nanmedian(bb_width[100:])  # Skip warmup period
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30  # 30% of capital
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    # Track MACD histogram for crossover detection
    prev_macd_hist = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(macd_hist[i]) or np.isnan(macd_hist[i-1]):
            continue
        
        if np.isnan(bb_width[i]):
            continue
        
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            continue
        
        # === 4H HMA TREND BIAS (Proven in #002) ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === MACD MOMENTUM ===
        # Histogram crossing above 0 = bullish momentum
        # Histogram crossing below 0 = bearish momentum
        macd_bull_cross = macd_hist[i] > 0 and prev_macd_hist <= 0
        macd_bear_cross = macd_hist[i] < 0 and prev_macd_hist >= 0
        
        # MACD histogram positive/negative
        macd_positive = macd_hist[i] > 0
        macd_negative = macd_hist[i] < 0
        
        # === BOLLINGER BAND VOLATILITY REGIME ===
        # BB Width < median = low vol (compression, breakout setup)
        # BB Width > median = high vol (trending)
        low_vol_regime = bb_width[i] < bb_width_median
        high_vol_regime = bb_width[i] > bb_width_median
        
        # === VOLUME CONFIRMATION ===
        # Volume > 1.5 * 20-period MA = confirmed move
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: MOMENTUM BREAKOUT (Low vol compression + MACD cross + Volume)
        if low_vol_regime:
            # Long: bullish 4H bias + MACD bull cross + volume confirmation
            if bull_bias and macd_bull_cross and volume_confirmed:
                new_signal = BASE_SIZE
            
            # Short: bearish 4H bias + MACD bear cross + volume confirmation
            elif bear_bias and macd_bear_cross and volume_confirmed:
                new_signal = -BASE_SIZE
        
        # MODE 2: TREND CONTINUATION (High vol + MACD aligned with trend)
        elif high_vol_regime:
            # Long: bullish 4H bias + MACD positive (momentum aligned)
            if bull_bias and macd_positive:
                new_signal = BASE_SIZE
            
            # Short: bearish 4H bias + MACD negative (momentum aligned)
            elif bear_bias and macd_negative:
                new_signal = -BASE_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit if 4H trend reverses against position
            if position_side > 0 and bear_bias:
                trend_exit = True
            if position_side < 0 and bull_bias:
                trend_exit = True
        
        # Apply stoploss or trend exit
        if stoploss_triggered or trend_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
        
        # Update previous MACD histogram for next iteration
        prev_macd_hist = macd_hist[i]
    
    return signals