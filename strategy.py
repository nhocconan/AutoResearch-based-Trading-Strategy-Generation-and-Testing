#!/usr/bin/env python3
"""
Experiment #151: 4h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + ATR Risk

Hypothesis: Previous 4h strategies failed due to over-filtering (too many confluence
requirements = 0 trades). Research shows HMA + RSI pullback works well on 4h timeframe
with proper HTF bias. This strategy uses:

1. 1d HMA(21) SLOPE: Major trend bias (only trade with HTF trend)
2. 4h HMA(16/48) crossover: Primary trend signal
3. RSI(14) pullback: Entry on retracement in trend direction
4. ATR(14) stoploss: 2.5x ATR trailing stop
5. 1w HMA: Ultra-long-term bias filter (avoid counter-trend to weekly)

Why this should work:
- HMA is faster than EMA with less lag (proven in literature)
- RSI pullback entries have higher win rate than breakouts
- 4h timeframe = 20-50 trades/year target (low fee drag)
- 1d + 1w HTF prevents fighting major trends
- SIMPLER logic = more trades generated (learned from 150 failures)

Timeframe: 4h (REQUIRED)
HTF: 1d + 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d1w_v1"
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

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 4h indicators
    hma_4h_16 = calculate_hma(close, 16)
    hma_4h_48 = calculate_hma(close, 48)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    HALF_SIZE = 0.15
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1W TREND BIAS (strongest filter) ===
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.2
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.2
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.3
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND SIGNAL ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === RSI PULLBACK CONDITIONS ===
        # Long: RSI pulled back to 40-50 in uptrend
        rsi_pullback_long = 35 < rsi_14[i] < 55
        # Short: RSI rallied to 45-60 in downtrend
        rsi_pullback_short = 45 < rsi_14[i] < 65
        
        # RSI extreme for counter-trend (less common)
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        # Reduce size if 1w and 1d disagree
        if (trend_1w_bullish and trend_1d_bearish) or (trend_1w_bearish and trend_1d_bullish):
            current_size = HALF_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - prioritize alignment
        long_confidence = 0
        
        # Path 1: All trends align bullish + RSI pullback (highest confidence)
        if trend_1w_bullish and trend_1d_bullish and hma_bullish and rsi_pullback_long:
            long_confidence = 3
        
        # Path 2: 1d + 4h bullish + RSI pullback
        elif trend_1d_bullish and hma_bullish and rsi_pullback_long:
            long_confidence = 2
        
        # Path 3: Price above 1w HMA + 4h bullish + RSI not overbought
        elif price_above_1w_hma and hma_bullish and rsi_14[i] < 60:
            long_confidence = 2
        
        # Path 4: RSI oversold + price above 1d HMA (dip buy)
        elif rsi_oversold and price_above_1d_hma:
            long_confidence = 1
        
        if long_confidence >= 2:
            new_signal = current_size
        elif long_confidence == 1 and bars_since_last_trade > 60:
            new_signal = HALF_SIZE
        
        # SHORT ENTRIES
        short_confidence = 0
        
        # Path 1: All trends align bearish + RSI pullback
        if trend_1w_bearish and trend_1d_bearish and hma_bearish and rsi_pullback_short:
            short_confidence = 3
        
        # Path 2: 1d + 4h bearish + RSI pullback
        elif trend_1d_bearish and hma_bearish and rsi_pullback_short:
            short_confidence = 2
        
        # Path 3: Price below 1w HMA + 4h bearish + RSI not oversold
        elif price_below_1w_hma and hma_bearish and rsi_14[i] > 40:
            short_confidence = 2
        
        # Path 4: RSI overbought + price below 1d HMA (rally sell)
        elif rsi_overbought and price_below_1d_hma:
            short_confidence = 1
        
        if short_confidence >= 2:
            new_signal = -current_size
        elif short_confidence == 1 and bars_since_last_trade > 60:
            new_signal = -HALF_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~20 days on 4h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 45:
                new_signal = HALF_SIZE
            elif trend_1d_bearish and rsi_14[i] > 55:
                new_signal = -HALF_SIZE
            elif rsi_14[i] < 25:
                new_signal = HALF_SIZE * 0.7
            elif rsi_14[i] > 75:
                new_signal = -HALF_SIZE * 0.7
        
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
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            if position_side < 0 and hma_bullish:
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