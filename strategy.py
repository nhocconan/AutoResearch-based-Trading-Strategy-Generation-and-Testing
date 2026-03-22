#!/usr/bin/env python3
"""
Experiment #208: 30m Primary + 4h/1d HTF — Simplified MTF Trend Pullback

Hypothesis: Previous 30m strategies failed due to over-complexity and too many conflicting
filters that resulted in 0 trades. This strategy uses a simpler, more reliable approach:

1. 4h HMA(21) for major trend direction (primary filter - don't fight the trend)
2. 1d HMA(21) slope for bias confirmation (avoid counter-trend in strong moves)
3. 30m RSI(7) for entry timing on pullbacks (less sensitive than CRSI, more trades)
4. Volume filter: only trade when volume > 0.7x 20-period average (lenient)
5. Choppiness Index: avoid trading when CHOP > 65 (too choppy)
6. ATR-based stoploss (2.0x ATR) with trailing

Key differences from failed strategies:
- Simpler entry logic (RSI pullback vs complex CRSI + Fisher + multiple regimes)
- Fewer conflicting filters (removed session filter that killed trades)
- More lenient thresholds to ensure 40-80 trades/year
- Discrete position sizing (0.25 base, max 0.35)
- Fallback mechanism if no trades for 250 bars

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_trend_pullback_4h1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Volume average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === 4H TREND DIRECTION ===
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 1D BIAS CONFIRMATION ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.15
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.15
        trend_1d_neutral = not trend_1d_bullish and not trend_1d_bearish
        
        # === CHOPPINESS FILTER ===
        is_choppy = chop_14[i] > 65
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * vol_avg_20[i]
        
        # === RSI PULLBACK (more lenient for trade generation) ===
        rsi_oversold = rsi_7[i] < 38
        rsi_overbought = rsi_7[i] > 62
        rsi_neutral_low = rsi_7[i] < 45
        rsi_neutral_high = rsi_7[i] > 55
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: 4h bullish + RSI pullback + volume + not choppy
        # Multiple paths to ensure trade generation
        long_score = 0
        
        # Path 1: Strong confluence (4h bullish + 1d bullish + RSI oversold)
        if price_above_4h_hma and trend_1d_bullish and rsi_oversold and volume_ok and not is_choppy:
            long_score = 3
        
        # Path 2: 4h bullish + RSI oversold + volume (1d neutral OK)
        elif price_above_4h_hma and rsi_oversold and volume_ok and not is_choppy:
            long_score = 2
        
        # Path 3: 4h bullish + 1d bullish + RSI moderate pullback
        elif price_above_4h_hma and trend_1d_bullish and rsi_neutral_low and volume_ok:
            long_score = 2
        
        # Path 4: Simple 4h bullish + deep RSI pullback
        elif price_above_4h_hma and rsi_7[i] < 30 and volume_ok:
            long_score = 2
        
        if long_score >= 2:
            new_signal = BASE_SIZE
        elif long_score == 1 and bars_since_last_trade > 100:
            new_signal = BASE_SIZE * 0.5
        
        # SHORT: 4h bearish + RSI rally + volume + not choppy
        short_score = 0
        
        # Path 1: Strong confluence
        if price_below_4h_hma and trend_1d_bearish and rsi_overbought and volume_ok and not is_choppy:
            short_score = 3
        
        # Path 2: 4h bearish + RSI overbought + volume
        elif price_below_4h_hma and rsi_overbought and volume_ok and not is_choppy:
            short_score = 2
        
        # Path 3: 4h bearish + 1d bearish + RSI moderate rally
        elif price_below_4h_hma and trend_1d_bearish and rsi_neutral_high and volume_ok:
            short_score = 2
        
        # Path 4: Simple 4h bearish + deep RSI rally
        elif price_below_4h_hma and rsi_7[i] > 70 and volume_ok:
            short_score = 2
        
        if short_score >= 2:
            new_signal = -BASE_SIZE
        elif short_score == 1 and bars_since_last_trade > 100:
            new_signal = -BASE_SIZE * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 250 bars (~5 days on 30m), force entry with relaxed conditions
        if bars_since_last_trade > 250 and new_signal == 0.0 and not in_position:
            if price_above_4h_hma and rsi_7[i] < 50 and volume_ok:
                new_signal = BASE_SIZE * 0.4
            elif price_below_4h_hma and rsi_7[i] > 50 and volume_ok:
                new_signal = -BASE_SIZE * 0.4
        
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
        
        if stoploss_triggered:
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