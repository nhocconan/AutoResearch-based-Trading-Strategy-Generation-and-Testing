#!/usr/bin/env python3
"""
Experiment #036: 12h KAMA-Choppiness Regime Switch with 1d Trend Bias

Hypothesis: Previous 12h strategies failed because they used single-regime logic
(either always trend-follow OR always mean-revert). This strategy ADAPTS:
- When CHOP > 61.8 (choppy/range): Use RSI mean-reversion at Bollinger bands
- When CHOP < 38.2 (trending): Use KAMA breakout with Donchian confirmation
- 1d HMA(21) provides major trend bias (only take trades in direction)

Why this should work better than #026/#032:
- KAMA adapts to volatility (faster in trends, slower in chop) vs static HMA
- Choppiness Index explicitly detects regime vs implicit via multiple filters
- Simpler entry logic = more trades (addressing 0-trade failure mode)
- Regime-switch means we profit in BOTH trending AND ranging markets

Key differences from failed strategies:
- NOT complex 3+ confluence (that killed #025, #030, #034)
- NOT pure mean-reversion (that killed #024, #027)
- Uses ADAPTIVE logic based on measured market state

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-50/year (12h natural frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_regime_1d_hma_atr_v1"
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
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    Faster in trends, slower in choppy markets.
    """
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close_s.diff(period))
    volatility = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum()
    er = change / volatility
    er = er.fillna(0)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]) or np.isnan(close_s.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    Measures whether market is trending or chopping.
    CHOP > 61.8 = choppy/range (mean-revert)
    CHOP < 38.2 = trending (trend-follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Sum of ATR over period
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    chop = chop.fillna(50)  # Default to neutral
    
    return chop.values

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper.values, lower.values, sma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return rsi.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_kama(df_1d['close'].values, period=10)  # KAMA on daily
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_12h_10 = calculate_kama(close, period=10)  # Fast KAMA
    kama_12h_30 = calculate_kama(close, period=30)  # Slow KAMA
    chop_14 = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(kama_12h_10[i]) or np.isnan(kama_12h_30[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H KAMA TREND ===
        kama_bullish = kama_12h_10[i] > kama_12h_30[i]
        kama_bearish = kama_12h_10[i] < kama_12h_30[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 61.8  # Range market - mean revert
        is_trending = chop_14[i] < 38.2  # Trend market - trend follow
        # Neutral zone: 38.2 - 61.8 (use trend logic as default)
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        
        if is_trending:
            # TREND REGIME: KAMA breakout + Donchian confirmation
            # Long: KAMA bullish + price breaks Donchian upper + daily bias bullish
            if kama_bullish and daily_bullish:
                if i > 0 and not np.isnan(donchian_upper[i-1]):
                    if close[i] > donchian_upper[i-1]:
                        new_signal = BASE_SIZE
            
            # Short: KAMA bearish + price breaks Donchian lower + daily bias bearish
            if kama_bearish and daily_bearish:
                if i > 0 and not np.isnan(donchian_lower[i-1]):
                    if close[i] < donchian_lower[i-1]:
                        new_signal = -BASE_SIZE
        
        elif is_choppy:
            # CHOPPY REGIME: RSI mean-reversion at Bollinger bands
            # Long: RSI < 30 + price at/near BB lower + daily bias NOT strongly bearish
            if rsi_14[i] < 30 and close[i] <= bb_lower[i]:
                if not daily_bearish or rsi_14[i] < 25:  # Allow long even if daily bearish at extreme RSI
                    new_signal = BASE_SIZE * 0.8  # Smaller size for mean-revert
            
            # Short: RSI > 70 + price at/near BB upper + daily bias NOT strongly bullish
            if rsi_14[i] > 70 and close[i] >= bb_upper[i]:
                if not daily_bullish or rsi_14[i] > 75:  # Allow short even if daily bullish at extreme RSI
                    new_signal = -BASE_SIZE * 0.8
        
        else:
            # NEUTRAL REGIME: Use simpler KAMA crossover with RSI filter
            # Long: KAMA bullish + RSI > 50 (momentum confirmation)
            if kama_bullish and rsi_14[i] > 50:
                new_signal = BASE_SIZE * 0.7
            
            # Short: KAMA bearish + RSI < 50
            if kama_bearish and rsi_14[i] < 50:
                new_signal = -BASE_SIZE * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 50 bars (~25 days on 12h), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            # Allow entry with just KAMA + daily alignment (no breakout/RSI extreme)
            if kama_bullish and daily_bullish and rsi_14[i] > 45:
                new_signal = BASE_SIZE * 0.6
            elif kama_bearish and daily_bearish and rsi_14[i] < 55:
                new_signal = -BASE_SIZE * 0.6
        
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
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        regime_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and is_choppy and rsi_14[i] > 60:
                # Long in choppy market, RSI rising - take profit
                regime_exit = True
            if position_side < 0 and is_choppy and rsi_14[i] < 40:
                # Short in choppy market, RSI falling - take profit
                regime_exit = True
        
        # Apply stoploss or regime exit
        if stoploss_triggered or regime_exit:
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
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            # else: Same direction, maintain position (no signal change = no fee)
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals