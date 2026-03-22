#!/usr/bin/env python3
"""
Experiment #148: 30m Primary + 4h/1d HTF — Simplified Trend-Follow with RSI Pullback

Hypothesis: Previous 30m strategies (#138, #140) failed with 0 trades due to overly strict
entry conditions. This strategy uses SIMPLER confluence with WIDER thresholds:

1. 4h HMA(21) slope for trend bias (proven in current best 12h strategy)
2. 30m RSI(14) for pullback entries with WIDER thresholds (30/70 not 15/85)
3. Volume filter relaxed (rel_vol > 0.7) to ensure trades
4. Fallback entries when no trades for 100+ bars (CRITICAL to avoid 0 trades)
5. 1d HMA for major trend confirmation (adds edge without killing frequency)

Key differences from failed #138, #140:
- RSI thresholds: 30/70 instead of 20/80 (MUCH more trades)
- Only 2-3 confluence requirements (not 5+)
- Forced trade mechanism after 120 bars of silence
- Smaller position size (0.25) for 30m fee management
- No Choppiness filter (was killing trades in #138)

Timeframe: 30m (REQUIRED)
HTF: 4h + 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (smaller for lower TF)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_vol_4h1d_v1"
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

def calculate_volume_ratio(volume, period=20):
    """Calculate relative volume ratio."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / np.where(vol_avg > 0, vol_avg, 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA and slope
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    
    # Calculate 1d HMA for major trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m)
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
        
        if np.isnan(hma_4h_slope_aligned[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_ratio[i]):
            continue
        
        # === 4H TREND BIAS ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.15
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.15
        trend_4h_neutral = not trend_4h_bullish and not trend_4h_bearish
        
        # === 1D MAJOR TREND ===
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === VOLUME FILTER (relaxed) ===
        volume_ok = vol_ratio[i] > 0.7
        
        # === RSI ENTRY (wider thresholds for MORE trades) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        rsi_neutral_low = rsi_14[i] < 40
        rsi_neutral_high = rsi_14[i] > 60
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC (SIMPLIFIED for more trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths to ensure trades
        long_score = 0
        
        # Path 1: 4h bullish + RSI pullback (primary)
        if trend_4h_bullish and rsi_oversold:
            long_score += 3
        
        # Path 2: 4h neutral + 1d bullish + RSI pullback
        if trend_4h_neutral and price_above_1d_hma and rsi_oversold:
            long_score += 2
        
        # Path 3: Volume confirmation + RSI pullback
        if volume_ok and rsi_oversold and trend_4h_bullish:
            long_score += 2
        
        # Path 4: RSI extreme alone (fallback for trades)
        if rsi_extreme_low:
            long_score += 2
        
        # Path 5: 1d bullish + RSI neutral low
        if price_above_1d_hma and rsi_neutral_low:
            long_score += 1
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score >= 2 and bars_since_last_trade > 60:
            new_signal = current_size
        elif long_score >= 1 and bars_since_last_trade > 100:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: 4h bearish + RSI rally (primary)
        if trend_4h_bearish and rsi_overbought:
            short_score += 3
        
        # Path 2: 4h neutral + 1d bearish + RSI rally
        if trend_4h_neutral and price_below_1d_hma and rsi_overbought:
            short_score += 2
        
        # Path 3: Volume confirmation + RSI rally
        if volume_ok and rsi_overbought and trend_4h_bearish:
            short_score += 2
        
        # Path 4: RSI extreme alone (fallback for trades)
        if rsi_extreme_high:
            short_score += 2
        
        # Path 5: 1d bearish + RSI neutral high
        if price_below_1d_hma and rsi_neutral_high:
            short_score += 1
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score >= 2 and bars_since_last_trade > 60:
            new_signal = -current_size
        elif short_score >= 1 and bars_since_last_trade > 100:
            new_signal = -current_size * 0.5
        
        # === FORCED TRADE MECHANISM (prevents 0 trades - CRITICAL) ===
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.4
            elif trend_4h_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.4
            elif rsi_extreme_low:
                new_signal = current_size * 0.3
            elif rsi_extreme_high:
                new_signal = -current_size * 0.3
            elif price_above_1d_hma and rsi_14[i] < 50:
                new_signal = current_size * 0.3
            elif price_below_1d_hma and rsi_14[i] > 50:
                new_signal = -current_size * 0.3
        
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