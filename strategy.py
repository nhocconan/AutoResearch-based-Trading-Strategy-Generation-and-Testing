#!/usr/bin/env python3
"""
Experiment #371: 12h KAMA Adaptive Trend with 1d HMA Bias + RSI Timing

Hypothesis: After 350+ failed experiments, the pattern is clear - over-filtered strategies
generate 0 trades. For 12h timeframe, I need SIMPLER logic that actually trades:

1. KAMA (Kaufman Adaptive Moving Average): Adapts to market noise automatically
   - Fast in trends, slow in ranges (no need for separate regime detection)
   - Period=10, fast=2, slow=30 (standard Kaufman parameters)
   - Crossover signals work better than static EMA on 12h

2. RSI(14) TIMING FILTER: Enter on pullbacks, not extremes
   - Long: KAMA bullish + RSI between 35-60 (pullback in uptrend)
   - Short: KAMA bearish + RSI between 40-65 (rally in downtrend)
   - NOT waiting for RSI<30 or >70 (too rare on 12h = 0 trades)

3. 1d HMA(21) SOFT BIAS: Weight entries, don't block
   - Long entries preferred when price > 1d HMA (but not required)
   - Short entries preferred when price < 1d HMA (but not required)
   - This is a WEIGHT, not a hard filter (generates more trades)

4. ATR(14) 2.5x TRAILING STOP: Protect capital
   - Signal → 0 when price moves 2.5*ATR against position
   - Trailing stop locks profits in strong trends

5. POSITION SIZING: 0.28 discrete (conservative for 12h)
   - Max 28% capital per position
   - Discrete levels: 0.0, ±0.28

Why this should work on 12h:
- KAMA adapts to volatility (no ADX filter needed = more trades)
- RSI range 35-65 is COMMON (generates 30-50 trades/year)
- Soft 1d bias doesn't block valid signals
- Simpler = more trades = better statistics

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_1d_hma_rsi_timing_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - fast in trends, slow in ranges.
    
    Efficiency Ratio (ER) = |close - close[n]| / sum(|close[i] - close[i-1]|)
    Smoothing Constant (SC) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < slow + period:
        return kama
    
    # Calculate Efficiency Ratio
    signal = np.abs(close - np.roll(close, period))
    signal[:period] = np.nan
    
    noise = np.abs(close - np.roll(close, 1))
    noise[0] = 0
    
    # Sum of noise over period
    noise_sum = pd.Series(noise).rolling(window=period, min_periods=period).sum().values
    
    # Efficiency Ratio (0 to 1)
    er = np.zeros(n)
    mask = noise_sum > 1e-10
    er[mask] = signal[mask] / noise_sum[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA with SMA of first period
    kama[period - 1] = np.nanmean(close[:period])
    
    # Calculate KAMA
    for i in range(period, n):
        if not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = close[i]
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (EMA with span=period)
    avg_gains = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_losses = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate RSI
    mask = avg_losses > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gains[mask] / avg_losses[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100  # No losses = RSI 100
    
    return rsi

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
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
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
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === KAMA TREND DIRECTION ===
        # KAMA slope: current vs 3 bars ago
        kama_bullish = kama[i] > kama[i - 3] if i >= 3 else False
        kama_bearish = kama[i] < kama[i - 3] if i >= 3 else False
        
        # Price vs KAMA position
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # === 1d HMA SOFT BIAS (not a hard filter) ===
        bull_bias_1d = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        bear_bias_1d = close[i] < hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        
        # === RSI TIMING FILTER (pullback entries, not extremes) ===
        # Long: RSI between 35-60 (pullback in uptrend, not oversold)
        rsi_long_ok = 35 <= rsi[i] <= 60
        
        # Short: RSI between 40-65 (rally in downtrend, not overbought)
        rsi_short_ok = 40 <= rsi[i] <= 65
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: KAMA bullish + RSI pullback + 1d bias helps
        if kama_bullish and price_above_kama and rsi_long_ok:
            # Prefer entries with 1d bullish bias (but not required)
            if bull_bias_1d:
                new_signal = SIZE
            else:
                # Weaker signal without 1d bias, but still enter
                new_signal = SIZE * 0.8
        
        # SHORT ENTRY: KAMA bearish + RSI rally + 1d bias helps
        elif kama_bearish and price_below_kama and rsi_short_ok:
            # Prefer entries with 1d bearish bias (but not required)
            if bear_bias_1d:
                new_signal = -SIZE
            else:
                # Weaker signal without 1d bias, but still enter
                new_signal = -SIZE * 0.8
        
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
        
        # === KAMA REVERSAL EXIT ===
        # Exit if KAMA trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and kama_bearish and price_below_kama:
                new_signal = 0.0
            if position_side < 0 and kama_bullish and price_above_kama:
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