#!/usr/bin/env python3
"""
Experiment #022: 12h KAMA Trend + RSI Momentum with 1d/1w Triple Bias

Hypothesis: Previous HMA-based strategies failed due to lag in choppy markets.
KAMA (Kaufman Adaptive Moving Average) adapts to volatility - fast in trends,
slow in ranges. Combined with triple HTF bias (1d + 1w), this should:
1. Filter out counter-trend trades in 2022 crash and 2025 bear
2. Generate more trades than pure breakout strategies (addressing 0-trade failures)
3. Use RSI momentum (not extreme pullback) for better entry timing
4. Choppiness Index to reduce trades in range markets (fee savings)

Key innovations vs failed experiments:
- KAMA instead of HMA (adaptive to volatility, less whipsaw)
- Triple HTF bias (1d + 1w) for stronger trend confirmation
- RSI momentum zone (40-60) instead of extreme pullback (more trades)
- Choppiness filter to avoid range chop (reduces fee drag)
- 2.0 ATR stoploss (tighter than 2.5, protects capital)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, discrete levels
Stoploss: 2.0 * ATR(14) trailing
Target: 20-50 trades/year, Sharpe > 0.028 (beat current best)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_momentum_1d1w_bias_v1"
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

def calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - fast during trends, slow during ranges.
    Based on Perry Kaufman's "Trading Systems and Methods".
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER): measures trend direction vs noise
    change = np.abs(close - np.roll(close, slow_period))
    change[0:slow_period] = np.abs(close[0:slow_period] - close[0])
    
    volatility = np.abs(close - np.roll(close, 1))
    volatility[0] = change[0]
    volatility_sum = pd.Series(volatility).rolling(window=slow_period, min_periods=slow_period).sum().values
    
    er = change / (volatility_sum + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    rsi = rsi.fillna(50).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    Based on E.W. Dreiss formula.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # ATR over period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # CHOP formula
    chop = 100 * np.log10((atr_sum + 1e-10) / (hh - ll + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    return upper.values, lower.values, middle.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1D indicators
    kama_1d_30 = calculate_kama(df_1d['close'].values, fast_period=2, slow_period=30, smoothing_period=10)
    
    # Calculate 1W indicators
    kama_1w_30 = calculate_kama(df_1w['close'].values, fast_period=2, slow_period=30, smoothing_period=10)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_30_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_30)
    kama_1w_30_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_30)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_12h_fast = calculate_kama(close, fast_period=2, slow_period=10, smoothing_period=5)
    kama_12h_slow = calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_middle = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.27
    
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
        
        if np.isnan(kama_1d_30_aligned[i]) or np.isnan(kama_1w_30_aligned[i]):
            continue
        
        if np.isnan(kama_12h_fast[i]) or np.isnan(kama_12h_slow[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > kama_1d_30_aligned[i]
        daily_bearish = close[i] < kama_1d_30_aligned[i]
        
        # === 1W TREND BIAS (stronger filter) ===
        weekly_bullish = close[i] > kama_1w_30_aligned[i]
        weekly_bearish = close[i] < kama_1w_30_aligned[i]
        
        # === 12H KAMA TREND ===
        kama_bullish = kama_12h_fast[i] > kama_12h_slow[i]
        kama_bearish = kama_12h_fast[i] < kama_12h_slow[i]
        
        # === CHOPPINESS REGIME FILTER ===
        # CHOP < 50 = trending (allow trades), CHOP > 61.8 = choppy (reduce size)
        is_trending = chop_14[i] < 50
        is_choppy = chop_14[i] > 61.8
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        # Reduce size in choppy markets
        if is_choppy:
            vol_adjustment *= 0.6
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.18, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: 12h KAMA bullish + 1d bias bullish + RSI momentum + trending regime
        # Require at least 1d bias, prefer 1w bias too
        if kama_bullish and daily_bullish:
            # RSI momentum zone (not extreme, just positive momentum)
            if 42 <= rsi_14[i] <= 65:
                # Prefer trending regime, but allow in neutral
                if is_trending or (not is_choppy):
                    # Extra confirmation: price above BB middle
                    if close[i] > bb_middle[i]:
                        new_signal = current_size
            
            # Breakout entry: price breaks above BB upper with momentum
            if close[i] > bb_upper[i] and rsi_14[i] > 50 and rsi_14[i] < 75:
                if is_trending:
                    new_signal = current_size
        
        # SHORT: 12h KAMA bearish + 1d bias bearish + RSI momentum + trending regime
        elif kama_bearish and daily_bearish:
            # RSI momentum zone (not extreme, just negative momentum)
            if 35 <= rsi_14[i] <= 58:
                # Prefer trending regime, but allow in neutral
                if is_trending or (not is_choppy):
                    # Extra confirmation: price below BB middle
                    if close[i] < bb_middle[i]:
                        new_signal = -current_size
            
            # Breakdown entry: price breaks below BB lower with momentum
            if close[i] < bb_lower[i] and rsi_14[i] < 50 and rsi_14[i] > 25:
                if is_trending:
                    new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 35 bars (~18 days on 12h), allow weaker entry
        if bars_since_last_trade > 35 and new_signal == 0.0 and not in_position:
            if kama_bullish and daily_bullish and rsi_14[i] > 45:
                new_signal = current_size * 0.7
            elif kama_bearish and daily_bearish and rsi_14[i] < 55:
                new_signal = -current_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and kama_bearish:
                trend_reversal = True
            if position_side < 0 and kama_bullish:
                trend_reversal = True
        
        # === DAILY BIAS REVERSAL EXIT ===
        # Exit if 1d bias flips against position
        bias_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and daily_bearish:
                bias_reversal = True
            if position_side < 0 and daily_bullish:
                bias_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or bias_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals