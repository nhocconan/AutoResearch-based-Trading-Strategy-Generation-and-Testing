#!/usr/bin/env python3
"""
Experiment #010: 1h Multi-Timeframe Trend Pullback with Regime Filter

Hypothesis: Previous 1h/30m strategies failed (0 trades) due to overly strict entry
conditions. This strategy uses PROVEN pattern from research:

1. 4h HMA(21) for PRIMARY trend direction (call ONCE before loop)
2. 12h HMA(21) for MAJOR trend bias (filter counter-trend)
3. 1h RSI(14) pullback entry within HTF trend (not extreme - RSI 40-60 range)
4. Choppiness Index(14) regime filter (only trade when CHOP 40-60 = transitional)
5. Volume filter (volume > 0.8x 20-bar avg) for confirmation
6. Session filter (8-20 UTC) for liquidity
7. ATR(14) 2.5x trailing stoploss

Why this should work on 1h:
- HTF (4h/12h) determines DIRECTION, 1h only for ENTRY TIMING
- RSI pullback (not extreme) generates MORE trades than CRSI extremes
- Choppiness filter avoids whipsaw in strong trends AND dead ranges
- Session filter reduces false breakouts in low-liquidity hours
- Discrete sizing (0.25/0.35) minimizes fee churn

Timeframe: 1h (REQUIRED for Experiment #010)
HTF: 4h and 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year (strict enough for fees, loose enough for generation)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h_12h_chop_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    """
    close_s = pd.Series(close)
    n = period
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    CHOP 40-60 = transitional (best for pullback entries)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - low_s.rolling(window=period, min_periods=period).min()
    
    chop = 100 * np.log10(atr_sum / hh_ll) / np.log10(period)
    chop = chop.fillna(50).values
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA for primary trend
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 12h HMA for major trend bias
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    # Additional 1h trend confirmation
    hma_1h_21 = calculate_hma(close, period=21)
    hma_1h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === 12H MAJOR TREND BIAS ===
        daily_bullish = close[i] > hma_12h_21_aligned[i]
        daily_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === 4H PRIMARY TREND ===
        hma_4h_bullish = close[i] > hma_4h_21_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 1H SHORT-TERM TREND ===
        hma_1h_bullish = hma_1h_21[i] > hma_1h_50[i]
        hma_1h_bearish = hma_1h_21[i] < hma_1h_50[i]
        
        # === CHOPPINESS REGIME ===
        chop_transitional = 40 <= chop_14[i] <= 60  # Best for pullbacks
        chop_trending = chop_14[i] < 45  # Strong trend
        chop_ranging = chop_14[i] > 55  # Range-bound
        
        # === RSI PULLBACK (not extreme - generates more trades) ===
        rsi_pullback_long = 35 <= rsi_14[i] <= 50  # Pullback in uptrend
        rsi_pullback_short = 50 <= rsi_14[i] <= 65  # Pullback in downtrend
        rsi_strong_long = rsi_14[i] < 40  # Stronger signal
        rsi_strong_short = rsi_14[i] > 60  # Stronger signal
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_avg_20[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 4h/12h bullish + RSI pullback + volume + session
        long_score = 0
        long_confidence = 0
        
        # Primary trigger: RSI pullback in bullish trend
        if hma_4h_bullish and hma_12h_bullish:
            if rsi_strong_long:
                long_score += 2.5
                long_confidence = 1
            elif rsi_pullback_long:
                long_score += 1.5
                long_confidence = 0.7
        
        # 1h trend confirmation
        if hma_1h_bullish:
            long_score += 1.0
        elif hma_1h_21[i] > close[i] * 0.98:  # Near support
            long_score += 0.5
        
        # Regime filter (transitional or trending works for pullbacks)
        if chop_transitional or chop_trending:
            long_score += 1.0
        
        # Volume confirmation
        if volume_confirmed:
            long_score += 0.5
        
        # Session filter (optional - relax for trade generation)
        if in_session:
            long_score += 0.3
        
        # Enter long if score >= 4.0 (3+ confluence)
        if long_score >= 4.0:
            new_signal = BASE_SIZE if long_confidence == 1 else REDUCED_SIZE
        
        # SHORT ENTRY: 4h/12h bearish + RSI pullback + volume + session
        short_score = 0
        short_confidence = 0
        
        # Primary trigger: RSI pullback in bearish trend
        if hma_4h_bearish and hma_12h_bearish:
            if rsi_strong_short:
                short_score += 2.5
                short_confidence = 1
            elif rsi_pullback_short:
                short_score += 1.5
                short_confidence = 0.7
        
        # 1h trend confirmation
        if hma_1h_bearish:
            short_score += 1.0
        elif hma_1h_21[i] < close[i] * 1.02:  # Near resistance
            short_score += 0.5
        
        # Regime filter
        if chop_transitional or chop_trending:
            short_score += 1.0
        
        # Volume confirmation
        if volume_confirmed:
            short_score += 0.5
        
        # Session filter
        if in_session:
            short_score += 0.3
        
        # Enter short if score >= 4.0 (3+ confluence)
        if short_score >= 4.0:
            new_signal = -BASE_SIZE if short_confidence == 1 else -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 120 bars (~5 days on 1h), allow weaker entry
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if hma_4h_bullish and rsi_14[i] < 45:
                new_signal = REDUCED_SIZE
            elif hma_4h_bearish and rsi_14[i] > 55:
                new_signal = -REDUCED_SIZE
        
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
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long if RSI goes overbought (momentum exhausted)
            if position_side > 0 and rsi_14[i] > 70:
                rsi_exit = True
            # Exit short if RSI goes oversold (momentum exhausted)
            if position_side < 0 and rsi_14[i] < 30:
                rsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend turns bearish
            if position_side > 0 and hma_4h_bearish:
                trend_reversal = True
            # Exit short if 4h trend turns bullish
            if position_side < 0 and hma_4h_bullish:
                trend_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or rsi_exit or trend_reversal:
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