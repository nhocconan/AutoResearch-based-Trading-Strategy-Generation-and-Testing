#!/usr/bin/env python3
"""
Experiment #094: 4h Primary + 12h/1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Previous strategies failed due to overly complex regime detection (CHOP + ADX + 
multiple confluence filters). This strategy uses PROVEN pattern from best performer 
(mtf_hma_rsi_zscore_v1 Sharpe=5.4) but simplified for 4h timeframe:

1. 12h HMA(21) slope for major trend bias (only trade with HTF trend)
2. 4h RSI(14) pullback entries (RSI<45 long in uptrend, RSI>55 short in downtrend)
3. 4h HMA(8/21) crossover for entry timing confirmation
4. ATR(14) 2.5x trailing stop for risk management
5. Position size: 0.30 discrete (conservative, avoids blowup in crashes)

Why this should work:
- Simpler = more trades = better statistics (avoids 0-trade failure)
- 4h timeframe naturally limits to 30-60 trades/year (fee-efficient)
- HTF trend filter prevents counter-trend trades in strong moves
- RSI pullback works in both bull and bear markets (unlike pure trend)
- Based on proven pattern that achieved Sharpe=5.4

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_12h_v3"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # HMA for trend confirmation on 4h
    hma_8 = calculate_hma(close, 8)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_8[i]) or np.isnan(hma_21[i]):
            continue
        
        # === 12H TREND BIAS (MAJOR) ===
        # HMA slope > 0.3 = bullish bias (prefer longs)
        # HMA slope < -0.3 = bearish bias (prefer shorts)
        # This is the PRIMARY filter - only trade with HTF trend
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.3
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.3
        trend_12h_neutral = not trend_12h_bullish and not trend_12h_bearish
        
        # Price vs 12h HMA for additional confirmation
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        hma_fast_above_slow = hma_8[i] > hma_21[i]
        hma_fast_below_slow = hma_8[i] < hma_21[i]
        hma_above_50 = hma_21[i] > hma_50[i]
        hma_below_50 = hma_21[i] < hma_50[i]
        
        # === RSI PULLBACK SIGNALS ===
        # In uptrend: buy when RSI pulls back to 35-50 range
        # In downtrend: sell when RSI rallies to 50-65 range
        rsi_oversold = rsi_14[i] < 45
        rsi_overbought = rsi_14[i] > 55
        rsi_neutral = 45 <= rsi_14[i] <= 55
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in neutral 12h trend
        if trend_12h_neutral:
            current_size = BASE_SIZE * 0.5
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - require 12h bullish OR neutral + 4h confirmation
        if trend_12h_bullish:
            # Strong 12h uptrend: buy RSI pullback
            if rsi_oversold and hma_fast_above_slow:
                new_signal = current_size
            # Also enter if price above 12h HMA and RSI moderate
            elif price_above_12h_hma and rsi_14[i] < 50 and hma_fast_above_slow:
                new_signal = current_size * 0.8
        elif trend_12h_neutral:
            # Neutral 12h: only enter with strong 4h confirmation
            if rsi_14[i] < 40 and hma_fast_above_slow and hma_above_50:
                new_signal = current_size * 0.6
            elif rsi_14[i] < 35 and hma_fast_above_slow:
                new_signal = current_size * 0.7
        
        # SHORT ENTRIES - require 12h bearish OR neutral + 4h confirmation
        if trend_12h_bearish:
            # Strong 12h downtrend: sell RSI rally
            if rsi_overbought and hma_fast_below_slow:
                new_signal = -current_size
            # Also enter if price below 12h HMA and RSI moderate
            elif price_below_12h_hma and rsi_14[i] > 50 and hma_fast_below_slow:
                new_signal = -current_size * 0.8
        elif trend_12h_neutral:
            # Neutral 12h: only enter with strong 4h confirmation
            if rsi_14[i] > 60 and hma_fast_below_slow and hma_below_50:
                new_signal = -current_size * 0.6
            elif rsi_14[i] > 65 and hma_fast_below_slow:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~20 days on 4h), allow weaker entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if trend_12h_bullish and rsi_14[i] < 50:
                new_signal = current_size * 0.5
            elif trend_12h_bearish and rsi_14[i] > 50:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Long: track highest price, stop below
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Short: track lowest price, stop above
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        # Exit if 12h trend reverses against position
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h becomes strongly bearish
            if position_side > 0 and trend_12h_bearish:
                trend_reversal = True
            # Exit short if 12h becomes strongly bullish
            if position_side < 0 and trend_12h_bullish:
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
                # Flip position
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