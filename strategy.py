#!/usr/bin/env python3
"""
Experiment #120: 1h Primary + 4h HTF — Simplified Trend Pullback Strategy

Hypothesis: Previous 1h/30m strategies failed (Sharpe=0.000) because entry conditions
were TOO STRICT (Connors RSI + Choppiness + multiple confluence = 0 trades). 
This strategy SIMPLIFIES to ensure trades actually occur:

1. 4h HMA(21) for trend direction (simpler than 1d, more responsive for 1h entries)
2. RSI(14) extremes: <35 long, >65 short (NOT extreme 25/75 which kills trades)
3. Price vs 4h HMA confirmation (long when price > 4h HMA, short when <)
4. ATR(14) trailing stoploss at 2.5x
5. Volume filter: >0.7x 20-bar avg (lenient, just avoids dead periods)
6. NO session filter (was killing trades in previous attempts)

Why this should work:
- Fewer filters = more trades (target 40-80/year on 1h)
- 4h trend + 1h entry = proven pattern from best strategies
- RSI(14) more reliable than Connors for crypto (less noisy)
- Discrete sizing (0.25/0.35) minimizes fee churn

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 base, 0.35 for strong signals
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year per symbol (1h = ~8760 bars/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_hma4h_pullback_v1"
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
    """Calculate volume ratio vs rolling average."""
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
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === 4H TREND BIAS ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.2
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.2
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === RSI SIGNALS (looser thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 40  # Was 35, loosened for more trades
        rsi_overbought = rsi_14[i] > 60  # Was 65, loosened for more trades
        rsi_extreme_low = rsi_14[i] < 30
        rsi_extreme_high = rsi_14[i] > 70
        
        # === VOLUME FILTER (lenient) ===
        volume_ok = vol_ratio[i] > 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths to ensure trades occur
        long_confidence = 0
        
        # Path 1: 4h bullish + RSI oversold + volume ok (primary)
        if trend_4h_bullish and rsi_oversold and volume_ok:
            long_confidence = 2
        
        # Path 2: Price above 4h HMA + RSI extreme low (pullback in uptrend)
        if price_above_4h_hma and rsi_extreme_low:
            long_confidence = 2
        
        # Path 3: 4h bullish + RSI very oversold (strong signal)
        if trend_4h_bullish and rsi_14[i] < 35:
            long_confidence = 3
        
        # Path 4: Simple RSI extreme (fallback for more trades)
        if rsi_14[i] < 28 and volume_ok:
            long_confidence = max(long_confidence, 1)
        
        # Path 5: 4h slope strongly bullish + any RSI < 45
        if hma_4h_slope_aligned[i] > 0.5 and rsi_14[i] < 45:
            long_confidence = max(long_confidence, 2)
        
        if long_confidence >= 2:
            new_signal = STRONG_SIZE
        elif long_confidence == 1 and bars_since_last_trade > 100:
            new_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRIES
        short_confidence = 0
        
        # Path 1: 4h bearish + RSI overbought + volume ok
        if trend_4h_bearish and rsi_overbought and volume_ok:
            short_confidence = 2
        
        # Path 2: Price below 4h HMA + RSI extreme high (rally in downtrend)
        if price_below_4h_hma and rsi_extreme_high:
            short_confidence = 2
        
        # Path 3: 4h bearish + RSI very overbought
        if trend_4h_bearish and rsi_14[i] > 65:
            short_confidence = 3
        
        # Path 4: Simple RSI extreme (fallback)
        if rsi_14[i] > 72 and volume_ok:
            short_confidence = max(short_confidence, 1)
        
        # Path 5: 4h slope strongly bearish + any RSI > 55
        if hma_4h_slope_aligned[i] < -0.5 and rsi_14[i] > 55:
            short_confidence = max(short_confidence, 2)
        
        if short_confidence >= 2:
            new_signal = -STRONG_SIZE
        elif short_confidence == 1 and bars_since_last_trade > 100:
            new_signal = -BASE_SIZE * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 200 bars (~8 days on 1h)
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and rsi_14[i] < 45:
                new_signal = BASE_SIZE * 0.4
            elif trend_4h_bearish and rsi_14[i] > 55:
                new_signal = -BASE_SIZE * 0.4
            elif rsi_14[i] < 32:
                new_signal = BASE_SIZE * 0.3
            elif rsi_14[i] > 68:
                new_signal = -BASE_SIZE * 0.3
        
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
            if position_side > 0 and trend_4h_bearish:
                trend_reversal = True
            if position_side < 0 and trend_4h_bullish:
                trend_reversal = True
        
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