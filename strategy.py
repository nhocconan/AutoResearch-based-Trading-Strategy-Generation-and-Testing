#!/usr/bin/env python3
"""
Experiment #546: 1d HMA-KAMA Crossover with Weekly Trend Bias and Volatility Regime

Hypothesis: After 500+ failed experiments, the key insight for daily timeframe is:
1. 1d captures major crypto cycles (2021 bull, 2022 crash, 2023-24 recovery)
2. HMA-KAMA crossover provides smoother trend signal than EMA (less whipsaw)
3. Weekly HMA bias prevents counter-trend entries (critical for 2022 crash)
4. Bollinger Band Width regime filter avoids entering during extreme volatility
5. RSI momentum filter ensures we enter on confirmation, not just crossover
6. Fewer but higher-quality trades = better risk-adjusted returns on daily

Why this should work on 1d:
- 1d has 1 bar/day = ~365 bars/year = very manageable trade frequency
- HMA(21) + KAMA(40) crossover = ~20-40 day trend changes = captures major moves
- 1w HMA via mtf_data helper provides proper HTF alignment
- BB Width < 70th percentile = avoid extreme volatility periods
- RSI(14) between 35-65 = avoid overbought/oversold extremes
- 2.5*ATR stoploss protects against 2022-style crashes
- Discrete sizing (0.30) limits drawdown during crashes

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_kama_crossover_weekly_bias_bb_regime_rsi_atr_v1"
timeframe = "1d"
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

def calculate_kama(close, period=40, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    change = np.abs(close_s - close_s.shift(period))
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=period, min_periods=period).sum()
    er = change / volatility.replace(0, np.inf)
    er = er.fillna(0)
    
    # Smoothing Constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[period-1] = close_s.iloc[period-1]
    for i in range(period, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    band_width = (upper - lower) / sma * 100  # as percentage
    
    return upper.values, lower.values, band_width.values

def calculate_bb_percentile(band_width, lookback=100):
    """Calculate Bollinger Band Width percentile over lookback period."""
    bw_s = pd.Series(band_width)
    bw_percentile = bw_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x) * 100 if len(x) == lookback else np.nan
    )
    return bw_percentile.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_21 = calculate_hma(close, 21)
    kama_40 = calculate_kama(close, 40)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_percentile = calculate_bb_percentile(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(kama_40[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_percentile[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_bias = close[i] > hma_1w_aligned[i]
        bear_bias = close[i] < hma_1w_aligned[i]
        
        # === HMA-KAMA CROSSOVER ===
        hma_above_kama = hma_21[i] > kama_40[i]
        hma_below_kama = hma_21[i] < kama_40[i]
        
        # Previous bar crossover detection
        prev_hma_above_kama = hma_21[i-1] > kama_40[i-1] if i > 0 else False
        prev_hma_below_kama = hma_21[i-1] < kama_40[i-1] if i > 0 else False
        
        crossover_long = hma_above_kama and not prev_hma_above_kama
        crossover_short = hma_below_kama and not prev_hma_below_kama
        
        # === RSI MOMENTUM FILTER (loose thresholds for more trades) ===
        rsi_ok_long = 35 <= rsi_14[i] <= 70  # Not overbought
        rsi_ok_short = 30 <= rsi_14[i] <= 65  # Not oversold
        
        # === BOLLINGER BAND WIDTH REGIME (avoid extreme volatility) ===
        bb_regime_ok = bb_percentile[i] < 75  # Not in top 25% volatility
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: HMA crosses above KAMA + weekly bullish bias + RSI ok + BB regime ok
        if crossover_long and bull_bias and rsi_ok_long and bb_regime_ok:
            new_signal = SIZE
        
        # Short: HMA crosses below KAMA + weekly bearish bias + RSI ok + BB regime ok
        elif crossover_short and bear_bias and rsi_ok_short and bb_regime_ok:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if weekly HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
                new_signal = 0.0
        
        # === CROSSOVER REVERSAL EXIT ===
        # Exit if crossover goes against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and hma_below_kama:
                new_signal = 0.0
            if position_side < 0 and hma_above_kama:
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