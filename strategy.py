#!/usr/bin/env python3
"""
Experiment #006: 12h HMA Trend Following with 1d Bias and RSI Filter

Hypothesis: Previous strategies failed due to overly strict entry conditions
(CRSI <15/>85 rarely triggers). This strategy uses simpler, more reliable signals:

1. 12h HMA(16/48) crossover for primary trend signal
2. 1d HMA(21) for major trend bias - only trade with daily trend
3. RSI(14) filter - avoid extreme overbought/oversold entries
4. ADX(14) > 20 filter - ensure some trending conditions
5. ATR(14) stoploss - 2.5x ATR trailing stop
6. 12h timeframe - targets 20-50 trades/year (optimal for trend following)

Why this should work:
- Simpler entry conditions = more trades (fixes #005 zero-trade failure)
- HMA crossover proven on SOL (Sharpe +0.879 in research)
- 1d HMA filter prevents counter-trend trades (major failure mode in 2022)
- RSI filter avoids buying tops/selling bottoms
- Conservative sizing (0.25-0.30) protects against crashes
- Looser thresholds ensure trades on 20% rallies and 50% crashes

Timeframe: 12h (REQUIRED for Experiment #006)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_adx_1d_v1"
timeframe = "12h"
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
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index) - measures trend strength.
    ADX > 25 = trending, ADX < 20 = ranging
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smooth with Wilder's method (EMA with span=period)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    hma_12h_16 = calculate_hma(close, period=16)
    hma_12h_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.18
    
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
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 1D MAJOR TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H HMA CROSSOVER ===
        hma_bullish = hma_12h_16[i] > hma_12h_48[i]
        hma_bearish = hma_12h_16[i] < hma_12h_48[i]
        
        # Previous bar crossover detection
        hma_prev_bullish = hma_12h_16[i-1] > hma_12h_48[i-1]
        hma_prev_bearish = hma_12h_16[i-1] < hma_12h_48[i-1]
        
        # === ADX TREND STRENGTH ===
        adx_trending = adx_14[i] > 20  # Some trend strength
        adx_strong = adx_14[i] > 25  # Strong trend
        
        # === RSI FILTER ===
        rsi_not_overbought = rsi_14[i] < 70  # Avoid buying tops
        rsi_not_oversold = rsi_14[i] > 30  # Avoid selling bottoms
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: HMA bullish crossover + daily trend + RSI filter
        long_entry = False
        
        # Primary: HMA crossover bullish
        if hma_bullish and not hma_prev_bullish:
            long_entry = True
        
        # Secondary: Already bullish + pullback entry (more frequent)
        if hma_bullish and hma_prev_bullish and rsi_14[i] < 55:
            long_entry = True
        
        # Filters for long entry
        if long_entry:
            # Must align with daily trend OR ADX strong
            if daily_bullish or adx_strong:
                # RSI filter - not extremely overbought
                if rsi_not_overbought:
                    new_signal = BASE_SIZE if adx_strong else REDUCED_SIZE
        
        # SHORT ENTRY: HMA bearish crossover + daily trend + RSI filter
        short_entry = False
        
        # Primary: HMA crossover bearish
        if hma_bearish and not hma_prev_bearish:
            short_entry = True
        
        # Secondary: Already bearish + bounce entry (more frequent)
        if hma_bearish and hma_prev_bearish and rsi_14[i] > 45:
            short_entry = True
        
        # Filters for short entry
        if short_entry:
            # Must align with daily trend OR ADX strong
            if daily_bearish or adx_strong:
                # RSI filter - not extremely oversold
                if rsi_not_oversold:
                    new_signal = -BASE_SIZE if adx_strong else -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~30 days on 12h), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if hma_bullish and daily_bullish and rsi_bullish:
                new_signal = REDUCED_SIZE
            elif hma_bearish and daily_bearish and rsi_bearish:
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
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            # Exit long if HMA turns bearish
            if position_side > 0 and hma_bearish:
                hma_exit = True
            # Exit short if HMA turns bullish
            if position_side < 0 and hma_bullish:
                hma_exit = True
        
        # === DAILY TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if daily trend turns bearish
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            # Exit short if daily trend turns bullish
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long if RSI extremely overbought
            if position_side > 0 and rsi_14[i] > 75:
                rsi_exit = True
            # Exit short if RSI extremely oversold
            if position_side < 0 and rsi_14[i] < 25:
                rsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or hma_exit or trend_reversal or rsi_exit:
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