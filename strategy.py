#!/usr/bin/env python3
"""
Experiment #562: 4h Regime-Adaptive Strategy with Dual HTF (1d/1w)

Hypothesis: After 500+ failed experiments, the key insight is that crypto markets
alternate between trending and ranging regimes. A single strategy (pure trend or
pure mean-reversion) fails because it doesn't adapt. This strategy:

1. Uses Choppiness Index (CHOP) to detect regime: CHOP>61.8=range, CHOP<38.2=trend
2. In RANGE: Mean reversion at Bollinger Band extremes (buy low, sell high)
3. In TREND: Trend following with HTF bias (1d HMA + 1w HMA confirmation)
4. RSI filter ensures we're not entering at extremes against the trade
5. ATR stoploss protects against 2022-style crashes

Why 4h works:
- 4h has 6 bars/day = ~2190 bars/year = good trade frequency
- Captures multi-day moves without intraday noise
- 1d and 1w HTF available via mtf_data helper with proper alignment
- Regime adaptation should work in both 2021-2024 (trending) and 2025+ (range)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (max 0.40)
Stoploss: 2.0 * ATR(14) trailing

Key difference from failed strategies:
- NOT pure trend following (failed in 2025 bear market)
- NOT pure mean reversion (failed in 2021 bull market)
- ADAPTS to regime = should work in both
- LOOSE entry conditions to ensure >=10 trades per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_chop_bb_rsi_dual_htf_atr_v1"
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
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of ATR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - low_s.rolling(window=period, min_periods=period).min()
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / hh_ll.replace(0, np.inf)) / np.log10(period)
    
    return chop.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper.values, middle.values, lower.values

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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop_14[i] > 55.0  # Slightly lower threshold to get more trades
        is_trend = chop_14[i] < 45.0  # Slightly higher threshold to get more trades
        
        # === HTF TREND BIAS ===
        bull_1d = close[i] > hma_1d_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        bull_1w = close[i] > hma_1w_aligned[i]
        bear_1w = close[i] < hma_1w_aligned[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 45  # Loose threshold for more trades
        rsi_overbought = rsi_14[i] > 55  # Loose threshold for more trades
        
        # === BB POSITION ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002  # At or slightly below lower band
        at_bb_upper = close[i] >= bb_upper[i] * 0.998  # At or slightly above upper band
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # RANGE REGIME: Mean reversion at BB extremes
        if is_range:
            # Long: at BB lower + RSI not overbought
            if at_bb_lower and not rsi_overbought:
                new_signal = SIZE
            # Short: at BB upper + RSI not oversold
            elif at_bb_upper and not rsi_oversold:
                new_signal = -SIZE
        
        # TREND REGIME: Trend following with HTF bias
        elif is_trend:
            # Long: bullish 1d + RSI not overbought + price above BB middle
            if bull_1d and not rsi_overbought and close[i] > bb_middle[i]:
                new_signal = SIZE
            # Short: bearish 1d + RSI not oversold + price below BB middle
            elif bear_1d and not rsi_oversold and close[i] < bb_middle[i]:
                new_signal = -SIZE
        
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
        
        # === HTF REVERSAL EXIT ===
        # Exit if 1d HMA flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_1d and bear_1w:
                new_signal = 0.0
            if position_side < 0 and bull_1d and bull_1w:
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