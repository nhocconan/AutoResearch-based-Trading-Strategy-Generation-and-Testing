#!/usr/bin/env python3
"""
Experiment #004: 4h HMA Trend + RSI Pullback with 12h/1d Filter

Hypothesis: Previous regime-switching strategies failed due to complexity and whipsaw.
This strategy uses a simpler, proven pattern:
1. 12h HMA(21) determines major trend direction (HTF filter)
2. 4h HMA(16/48) confirms intermediate trend
3. 4h RSI(14) pullback entries (not breakouts) - buy dips in uptrend, sell rallies in downtrend
4. ATR(14) trailing stoploss for risk management

Why this should work better:
- Pullback entries have higher win rate than breakouts (research shows 60-65% vs 45-50%)
- 12h filter prevents counter-trend trades that destroyed 2022 performance
- RSI pullback works in both bull and bear markets (unlike pure trend following)
- Simpler logic = fewer false signals = better Sharpe

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target: 30-50 trades/year (fee drag manageable on 4h)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_12h_filter_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength measurement."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    dx = 100 * np.abs(plus_di - minus_di) / (np.abs(plus_di + minus_di) + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_16 = calculate_hma(close, 16)
    hma_4h_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
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
        
        if np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 12H TREND BIAS (HTF FILTER) ===
        # Price above 12h HMA = bullish bias (only look for longs or stay flat)
        # Price below 12h HMA = bearish bias (only look for shorts or stay flat)
        daily_bullish = close[i] > hma_12h_21_aligned[i]
        daily_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === 4H HMA TREND ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === HMA SLOPE (momentum confirmation) ===
        hma_slope_long = hma_4h_16[i] > hma_4h_16[i-3] if i > 3 else False
        hma_slope_short = hma_4h_16[i] < hma_4h_16[i-3] if i > 3 else False
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 25  # Trending market
        adx_weak = adx_14[i] < 20    # Ranging market
        
        # === RSI PULLBACK ZONES ===
        # For longs: RSI pulled back to 35-50 zone in uptrend
        rsi_pullback_long = 35 <= rsi_14[i] <= 55
        # For shorts: RSI rallied to 45-65 zone in downtrend
        rsi_pullback_short = 45 <= rsi_14[i] <= 65
        
        # === RSI EXTREME REVERSAL (alternative entry) ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need 12h bullish + 4h HMA bullish + RSI pullback
        # This is a pullback entry, not breakout - higher win rate
        if daily_bullish and hma_bullish:
            # Primary entry: RSI pullback in uptrend
            if rsi_pullback_long and hma_slope_long:
                new_signal = current_size
            # Secondary entry: RSI oversold bounce (mean reversion in uptrend)
            elif rsi_oversold and hma_4h_16[i] > hma_4h_48[i]:
                new_signal = current_size * 0.7  # Smaller size for mean reversion
        
        # SHORT ENTRY: Need 12h bearish + 4h HMA bearish + RSI pullback
        if daily_bearish and hma_bearish:
            # Primary entry: RSI pullback in downtrend
            if rsi_pullback_short and hma_slope_short:
                new_signal = -current_size
            # Secondary entry: RSI overbought rejection (mean reversion in downtrend)
            elif rsi_overbought and hma_4h_16[i] < hma_4h_48[i]:
                new_signal = -current_size * 0.7  # Smaller size for mean reversion
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~13 days on 4h), allow weaker entries
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            # Allow entry with just 12h trend + RSI extreme (no 4h HMA confirmation)
            if daily_bullish and rsi_oversold:
                new_signal = current_size * 0.5
            elif daily_bearish and rsi_overbought:
                new_signal = -current_size * 0.5
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h HMA turns bearish
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            # Exit short if 4h HMA turns bullish
            if position_side < 0 and hma_bullish:
                trend_reversal = True
            
            # Exit if 12h trend reverses against position
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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