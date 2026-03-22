#!/usr/bin/env python3
"""
Experiment #549: 1h KAMA Adaptive Trend with 4h HMA Bias and BB Regime Filter

Hypothesis: After 548 failed experiments, key insights are:
1. KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency ratio - works better in crypto than fixed EMA/HMA
2. 1h timeframe balances noise reduction vs trade frequency (15m/30m too noisy, 4h/12h too slow)
3. 4h HMA trend bias proven in successful strategies (mtf_4h_regime_chop_1d_1w_hma)
4. Bollinger Band Width percentile detects regime better than CHOP (CHOP failed in exp#537,539,541,542,544,545)
5. RSI for entry timing only, NOT primary signal (Connors RSI failed badly in exp#537,544)
6. Asymmetric sizing: larger positions in trending regime, smaller in ranging

Why this should work on 1h:
- KAMA automatically slows in chop, speeds in trends - no manual regime switch needed
- 4h HMA provides proven HTF trend filter (used in current best strategy)
- BB Width < 20th percentile = squeeze (prepare for breakout), > 80th = expansion (trend following)
- RSI 40-60 zone filter avoids extreme entries that failed in mean-reversion strategies
- 2*ATR stoploss protects against 2022-style crashes while allowing room for noise

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete (max 0.40)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_adaptive_4h_hma_bb_regime_rsi_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency - slows in chop, speeds in trends.
    Formula: KAMA = KAMA_prev + SC * (Price - KAMA_prev)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    ER = |Close - Close_n| / Sum(|Close_i - Close_i-1|)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    price_change = np.abs(close_s - close_s.shift(er_period))
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=er_period, min_periods=er_period).sum()
    er = price_change / volatility.replace(0, np.inf)
    er = er.fillna(0)
    
    # Smoothing Constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma  # Normalized band width
    
    return upper.values, lower.values, width.values

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
    
    return rsi.values

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate percentile rank of BB Width over lookback period."""
    bb_width_s = pd.Series(bb_width)
    percentile = bb_width_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x), raw=False
    )
    return percentile.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, 100)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_TREND = 0.30   # Larger position in trending regime
    SIZE_RANGE = 0.20   # Smaller position in ranging regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_1h[i]) or np.isnan(bb_width_pct[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS (proven HTF filter) ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === 1H KAMA TREND ===
        kama_bull = close[i] > kama_1h[i]
        kama_bear = close[i] < kama_1h[i]
        
        # === BB WIDTH REGIME DETECTION ===
        # Low percentile = squeeze (prepare for breakout)
        # High percentile = expansion (trend following)
        squeeze_regime = bb_width_pct[i] < 0.20  # Bottom 20% = squeeze
        expansion_regime = bb_width_pct[i] > 0.80  # Top 20% = expansion
        
        # === RSI ENTRY FILTER (avoid extremes that failed in mean-rev strategies) ===
        rsi_neutral = 40 <= rsi_14[i] <= 60  # Avoid RSI extremes
        rsi_bull = rsi_14[i] > 45  # Mild bullish momentum
        rsi_bear = rsi_14[i] < 55  # Mild bearish momentum
        
        # === DETERMINE POSITION SIZE BASED ON REGIME ===
        current_size = SIZE_TREND if expansion_regime else SIZE_RANGE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: KAMA bullish + 4H HMA bullish + RSI confirmation
        # Enter on squeeze (breakout prep) OR expansion (trend following)
        if kama_bull and bull_bias and rsi_bull:
            if squeeze_regime or expansion_regime:
                new_signal = current_size
        
        # Short: KAMA bearish + 4H HMA bearish + RSI confirmation
        if kama_bear and bear_bias and rsi_bear:
            if squeeze_regime or expansion_regime:
                new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4H HMA flips against position (stronger signal than 1H KAMA flip)
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
                new_signal = 0.0
        
        # === KAMA FLIP EXIT ===
        # Exit if 1H KAMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and kama_bear:
                new_signal = 0.0
            if position_side < 0 and kama_bull:
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