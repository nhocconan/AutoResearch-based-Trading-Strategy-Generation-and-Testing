#!/usr/bin/env python3
"""
Experiment #072: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + Volatility Regime

Hypothesis: Previous 12h strategies failed because HMA doesn't adapt to volatility regimes.
KAMA (Kaufman Adaptive Moving Average) adjusts smoothing based on market efficiency ratio,
performing better in ranging markets while capturing trends. Combined with:

1. KAMA(10/30) crossover for adaptive trend following (not fixed HMA)
2. ATR Ratio (ATR7/ATR30) for volatility regime detection (expansion vs contraction)
3. Bollinger Band Width squeeze for breakout confirmation
4. 1d KAMA slope + 1w price position for major trend context
5. Asymmetric entry: long only when 1w bullish, short only when 1w bearish
6. RSI(14) 45/55 thresholds (not extreme) for momentum confirmation
7. Position size: 0.28 discrete, reduced to 0.20 in low volatility
8. Stoploss: 2.5 * ATR(14) trailing (wider for 12h to avoid whipsaws)

Why this should beat previous attempts:
- KAMA adapts to market conditions (unlike fixed HMA/EMA)
- Volatility regime filter prevents entries during compression
- BB Width squeeze catches breakouts before they happen
- 1w HTF ensures we don't trade against major trend
- Asymmetric logic reduces counter-trend losses in bear markets
- Looser RSI thresholds (45/55 vs 20/80) ensure trade generation

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.28 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-45/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_volregime_bb_1d1w_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    ER = |net change| / sum of absolute changes over period
    SC = (ER * (fast_SC - slow_SC) + slow_SC)^2
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    net_change = np.abs(close_s.diff(er_period))
    sum_changes = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = net_change / (sum_changes + 1e-10)
    er = er.fillna(0).values
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize with price
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    bandwidth = (upper - lower) / sma * 100  # Normalized bandwidth
    
    return upper.values, lower.values, bandwidth.values

def calculate_kama_slope(kama_values, lookback=5):
    """Calculate KAMA slope over lookback period."""
    slope = np.zeros(len(kama_values))
    for i in range(lookback, len(kama_values)):
        if kama_values[i - lookback] != 0:
            slope[i] = (kama_values[i] - kama_values[i - lookback]) / kama_values[i - lookback] * 100
    return slope

def calculate_atr_ratio(atr_short, atr_long):
    """Calculate ATR ratio for volatility regime detection."""
    ratio = atr_short / (atr_long + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    kama_1d_20 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=20)
    kama_1d_slope = calculate_kama_slope(kama_1d_20, 5)
    
    # Calculate 1w HTF indicators
    kama_1w_20 = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=20)
    price_1w = df_1w['close'].values
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_20_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_20)
    kama_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_slope)
    kama_1w_20_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_20)
    price_1w_aligned = align_htf_to_ltf(prices, df_1w, price_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    
    # KAMA for trend (adaptive)
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=10)
    kama_30 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Bollinger Bands for squeeze detection
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_width_sma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    
    # ATR ratio for volatility regime
    atr_ratio = calculate_atr_ratio(atr_7, atr_30)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.20
    
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
        
        if np.isnan(kama_1d_20_aligned[i]) or np.isnan(kama_1d_slope_aligned[i]):
            continue
        
        if np.isnan(kama_1w_20_aligned[i]) or np.isnan(price_1w_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_10[i]) or np.isnan(kama_30[i]):
            continue
        
        if np.isnan(bb_width[i]) or np.isnan(bb_width_sma[i]):
            continue
        
        if np.isnan(atr_ratio[i]):
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        # Price above 1w KAMA = bullish major trend (prefer longs only)
        # Price below 1w KAMA = bearish major trend (prefer shorts only)
        trend_1w_bullish = close[i] > kama_1w_20_aligned[i]
        trend_1w_bearish = close[i] < kama_1w_20_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        # 1d KAMA slope > 0 = bullish intermediate trend
        # 1d KAMA slope < 0 = bearish intermediate trend
        trend_1d_bullish = kama_1d_slope_aligned[i] > 0.5  # Threshold to avoid noise
        trend_1d_bearish = kama_1d_slope_aligned[i] < -0.5
        
        # === 12H KAMA CROSSOVER (ADAPTIVE) ===
        # Fast KAMA(10) crosses above Slow KAMA(30) = bullish crossover
        # Fast KAMA(10) crosses below Slow KAMA(30) = bearish crossover
        kama_bullish_cross = kama_10[i] > kama_30[i] and kama_10[i-1] <= kama_30[i-1]
        kama_bearish_cross = kama_10[i] < kama_30[i] and kama_10[i-1] >= kama_30[i-1]
        
        # Current KAMA alignment
        kama_aligned_bullish = kama_10[i] > kama_30[i]
        kama_aligned_bearish = kama_10[i] < kama_30[i]
        
        # === VOLATILITY REGIME ===
        # ATR ratio > 1.2 = volatility expanding (good for breakouts)
        # ATR ratio < 0.8 = volatility contracting (wait for breakout)
        # ATR ratio 0.8-1.2 = normal volatility
        vol_expanding = atr_ratio[i] > 1.2
        vol_contracting = atr_ratio[i] < 0.8
        vol_normal = not vol_expanding and not vol_contracting
        
        # === BOLLINGER BAND SQUEEZE ===
        # BB Width < 80% of 50-period average = squeeze (potential breakout)
        bb_squeeze = bb_width[i] < 0.8 * bb_width_sma[i]
        bb_expansion = bb_width[i] > bb_width_sma[i]  # Bandwidth expanding
        
        # === RSI MOMENTUM ===
        # RSI > 50 = bullish momentum
        # RSI < 50 = bearish momentum
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in low volatility (less opportunity)
        if vol_contracting:
            current_size = REDUCED_SIZE
        
        # === ENTRY LOGIC (ASYMMETRIC) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (only when 1w bullish bias)
        if trend_1w_bullish:
            # Strong entry: KAMA crossover + vol expansion + RSI confirmation
            if kama_bullish_cross and (vol_expanding or vol_normal) and rsi_bullish:
                if trend_1d_bullish:  # 1d confirms
                    new_signal = current_size
            # Pullback entry in established uptrend
            elif kama_aligned_bullish and trend_1d_bullish:
                if rsi_14[i] > 45 and rsi_14[i] < 60:  # Pullback zone
                    if bb_expansion:  # Bands expanding after squeeze
                        new_signal = current_size * 0.8
        
        # SHORT ENTRIES (only when 1w bearish bias)
        if trend_1w_bearish:
            # Strong entry: KAMA crossover + vol expansion + RSI confirmation
            if kama_bearish_cross and (vol_expanding or vol_normal) and rsi_bearish:
                if trend_1d_bearish:  # 1d confirms
                    new_signal = -current_size
            # Pullback entry in established downtrend
            elif kama_aligned_bearish and trend_1d_bearish:
                if rsi_14[i] > 40 and rsi_14[i] < 55:  # Pullback zone
                    if bb_expansion:  # Bands expanding after squeeze
                        new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 120 bars (~60 days on 12h), allow weaker entry
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and kama_aligned_bullish and rsi_14[i] > 48:
                new_signal = current_size * 0.5
            elif trend_1w_bearish and kama_aligned_bearish and rsi_14[i] < 52:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1w trend reverses bearish
            if position_side > 0 and trend_1w_bearish and kama_aligned_bearish:
                trend_reversal = True
            # Exit short if 1w trend reverses bullish
            if position_side < 0 and trend_1w_bullish and kama_aligned_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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