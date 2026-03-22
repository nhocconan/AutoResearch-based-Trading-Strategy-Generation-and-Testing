#!/usr/bin/env python3
"""
Experiment #038: 30m Primary + 4h/1d HTF — Simplified Mean Reversion with Trend Filter

Hypothesis: Previous 30m strategies failed due to overly complex confluence filters
resulting in 0 trades. This strategy SIMPLIFIES entry conditions while keeping
HTF trend filter for direction control.

Key Changes from Failed Experiments:
1. Standard RSI(14) instead of Connors RSI (simpler, more reliable)
2. Bollinger Band %B for mean reversion entry (proven edge)
3. Relaxed volume filter (0.5x avg instead of 0.8x)
4. No session filter (crypto trades 24/7, session filter kills trade count)
5. Simpler position tracking logic
6. Looser RSI thresholds (35/65 instead of 20/80) to ensure trades

Entry Logic:
- LONG: 4h HMA bullish + 30m RSI < 35 + price < BB lower band
- SHORT: 4h HMA bearish + 30m RSI > 65 + price > BB upper band

Exit Logic:
- RSI crosses back through 50 (mean reversion complete)
- ATR trailing stoploss at 2.5x
- HTF trend reversal

Timeframe: 30m (REQUIRED)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (smaller for lower TF)
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_bb_rsi_4h1d_hma_simp_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and %B indicator."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # %B = (price - lower) / (upper - lower)
    bb_range = upper - lower
    bb_range[bb_range == 0] = 1e-10  # avoid division by zero
    percent_b = (close - lower) / bb_range
    
    return upper, lower, percent_b

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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_pct_b = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volume SMA for filter (relaxed threshold)
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100  # Allow immediate entry if conditions met
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_pct_b[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4H TREND DIRECTION (PRIMARY FILTER) ===
        # Price above 4h HMA = bullish (prefer longs)
        # Price below 4h HMA = bearish (prefer shorts)
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 1D TREND CONFIRMATION (SECONDARY FILTER) ===
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === VOLUME FILTER (relaxed) ===
        volume_ok = volume[i] > 0.5 * volume_sma[i]
        
        # === BOLLINGER %B EXTREMES ===
        # %B < 0 = price below lower band (oversold)
        # %B > 1 = price above upper band (overbought)
        bb_oversold = bb_pct_b[i] < 0.1
        bb_overbought = bb_pct_b[i] > 0.9
        
        # === RSI EXTREMES (loosened for trade frequency) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === RSI EXIT SIGNALS ===
        rsi_neutral = 45 < rsi_14[i] < 55
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 4h bullish + RSI oversold + BB oversold + volume
        # Loosen: require EITHER 4h OR 1d bullish (not both)
        if trend_4h_bullish or trend_1d_bullish:
            if rsi_oversold and bb_oversold and volume_ok:
                new_signal = BASE_SIZE
        
        # SHORT ENTRY: 4h bearish + RSI overbought + BB overbought + volume
        if trend_4h_bearish or trend_1d_bearish:
            if rsi_overbought and bb_overbought and volume_ok:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~6 days on 30m), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and rsi_14[i] < 40:
                new_signal = BASE_SIZE * 0.6
            elif trend_4h_bearish and rsi_14[i] > 60:
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
        
        # === MEAN REVERSION EXIT ===
        # Exit when RSI returns to neutral (mean reversion complete)
        mean_reversion_exit = False
        if in_position and position_side != 0:
            if rsi_neutral:
                mean_reversion_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_4h_bearish and rsi_14[i] > 55:
                trend_reversal = True
            if position_side < 0 and trend_4h_bullish and rsi_14[i] < 45:
                trend_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or mean_reversion_exit or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New position
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
            # If same direction, keep position (no update needed)
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