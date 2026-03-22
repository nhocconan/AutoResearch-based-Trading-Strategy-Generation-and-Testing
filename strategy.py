#!/usr/bin/env python3
"""
Experiment #191: 12h KAMA Trend + 1d/1w HMA Filter + BB Width Regime + RSI Pullback

Hypothesis: 12h timeframe with KAMA (Kaufman Adaptive Moving Average) provides better
trend adaptation than HMA or EMA. KAMA speeds up in trends and slows in ranges,
reducing whipsaw. Adding 1d AND 1w HMA filters ensures we only trade with the
multi-timeframe trend. Bollinger Band Width regime filter avoids low-volatility
traps. RSI pullback entries (RSI 40-60 in uptrend, 40-60 in downtrend) catch
retracements rather than chasing breakouts.

Why this might work better than #179:
- #179 used Donchian breakouts which chase momentum (whipsaw in ranges)
- KAMA adapts to volatility automatically (no ADX threshold needed)
- Dual HTF filter (1d + 1w) = stronger trend confirmation
- RSI pullback entries = better risk/reward than breakout entries
- BB Width filter = avoid trading during compression (false breakouts)
- More lenient conditions = ensure enough trades (learning from #190 zero trades)

Learning from failures:
- #179 (12h Donchian): Sharpe=-0.319 - breakout chasing failed
- #185 (12h Chop regime): Sharpe=-0.299 - choppiness filter didn't help
- #190 (4h Fisher): Sharpe=0.000, 0 trades - conditions too strict
- Mean reversion alone fails, but pullback IN trend works
- Need BOTH 1d AND 1w alignment for strong trend bias

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete (conservative for 12h swings)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_1d_1w_hma_bb_rsi_pullback_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - fast in trends, slow in ranges.
    Efficiency Ratio (ER) determines smoothing constant.
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = change[period]
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    volatility[0:period] = volatility[period] if period < n else 1.0
    
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = change / volatility
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bb(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma  # Normalized band width
    
    return upper, lower, width

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

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

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
    kama = calculate_kama(close, period=10)
    bb_upper, bb_lower, bb_width = calculate_bb(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Calculate BB Width percentile for regime (rolling 100 periods)
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x < x[-1]) / len(x), raw=True
    ).values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.28
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(bb_width[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # Both 1d AND 1w must agree for strong trend signal
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bull: both 1d and 1w bullish
        # Strong bear: both 1d and 1w bearish
        # Weak trend: only one timeframe agrees (still tradeable with smaller size)
        strong_bull = bull_trend_1d and bull_trend_1w
        strong_bear = bear_trend_1d and bear_trend_1w
        weak_bull = bull_trend_1d or bull_trend_1w
        weak_bear = bear_trend_1d or bear_trend_1w
        
        # === BOLLINGER BAND WIDTH REGIME ===
        # BB Width in bottom 30% = squeeze (low vol, potential breakout)
        # BB Width in top 70% = expansion (high vol, trend continuation)
        bb_squeeze = bb_width_percentile[i] < 0.30 if not np.isnan(bb_width_percentile[i]) else False
        bb_expansion = bb_width_percentile[i] > 0.70 if not np.isnan(bb_width_percentile[i]) else False
        
        # === KAMA TREND ===
        # Price above KAMA = bullish, below = bearish
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # KAMA slope (compare to 3 bars ago)
        kama_slope_bull = kama[i] > kama[i-3] if i >= 3 else False
        kama_slope_bear = kama[i] < kama[i-3] if i >= 3 else False
        
        # === RSI PULLBACK ENTRY ===
        # In uptrend: enter on RSI pullback to 40-55 (not overbought)
        # In downtrend: enter on RSI bounce to 45-60 (not oversold)
        rsi_pullback_long = 35 < rsi[i] < 60
        rsi_pullback_short = 40 < rsi[i] < 65
        
        # === EMA CONFIRMATION ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: Strong/weak bull trend + KAMA bullish + RSI pullback + (BB expansion OR EMA bull)
        # More flexible than breakout - catch pullbacks in trend
        if (strong_bull or weak_bull) and kama_bullish and kama_slope_bull:
            if rsi_pullback_long:
                # Need at least one confirmation
                if bb_expansion or ema_bullish or bb_squeeze:
                    new_signal = SIZE_BASE
        
        # Short: Strong/weak bear trend + KAMA bearish + RSI pullback + (BB expansion OR EMA bear)
        if (strong_bear or weak_bear) and kama_bearish and kama_slope_bear:
            if rsi_pullback_short:
                # Need at least one confirmation
                if bb_expansion or ema_bearish or bb_squeeze:
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