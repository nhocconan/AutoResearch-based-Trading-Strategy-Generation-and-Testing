#!/usr/bin/env python3
"""
Experiment #006: 12h HMA Trend + Donchian Breakout + 1d Trend Filter

Hypothesis: Previous strategies failed due to over-complexity (CRSI+Chop) or 
too few trades (1d timeframe). This strategy uses:

1. HMA(16/48) crossover - proven trend indicator with less lag than EMA
2. Donchian(20) breakout - clean entry signal for crypto trends
3. Choppiness Index(14) - regime filter (trend when CHOP<38.2, range when >61.8)
4. 1d HMA(21) trend filter via mtf_data - only trade with daily trend
5. RSI(14) momentum - confirms breakout direction
6. ATR(14) 2.5x trailing stop - protects against reversals

Why 12h should work:
- Higher TF = fewer false signals, less fee drag
- 20-50 trades/year target (optimal for crypto)
- 1d filter prevents counter-trend trades
- Simpler logic = more trades (avoids 0-trade failure mode)

Timeframe: 12h (REQUIRED for Experiment #006)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.30 high conviction
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_donchian_chop_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - reduces lag while maintaining smoothness."""
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel highs and lows."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    donchian_high = high_s.rolling(window=period, min_periods=period).max().values
    donchian_low = low_s.rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    return donchian_high, donchian_low, donchian_mid

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    Formula: 100 * (ATR(1) sum) / (Highest High - Lowest Low) / sqrt(period)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over period
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100 * (tr_sum / hh_ll) / np.sqrt(period)
    chop = chop.replace([np.inf, -np.inf], np.nan)
    
    return chop.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    
    return rsi.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend filter
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    hma_12h_16 = calculate_hma(close, period=16)
    hma_12h_48 = calculate_hma(close, period=48)
    
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    
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
        
        if np.isnan(chop_14[i]) or np.isnan(donchian_high[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        # === DAILY TREND FILTER ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === HMA TREND (12h) ===
        hma_bullish = hma_12h_16[i] > hma_12h_48[i]
        hma_bearish = hma_12h_16[i] < hma_12h_48[i]
        
        # === CHOPPINESS REGIME ===
        trending_regime = chop_14[i] < 38.2  # Trending market
        ranging_regime = chop_14[i] > 61.8   # Range/choppy market
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = False
        donchian_breakout_short = False
        
        if i > 0:
            # Long: price breaks above Donchian high
            if close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]:
                donchian_breakout_long = True
            # Short: price breaks below Donchian low
            if close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]:
                donchian_breakout_short = True
        
        # Continuation signals
        above_donchian = close[i] > donchian_high[i]
        below_donchian = close[i] < donchian_low[i]
        
        # === RSI MOMENTUM ===
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        rsi_strong_bull = rsi_14[i] > 55
        rsi_strong_bear = rsi_14[i] < 45
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY - requires trend alignment + momentum
        long_score = 0
        
        # HMA trend (primary)
        if hma_bullish:
            long_score += 2
        
        # Daily trend filter (must align)
        if daily_bullish:
            long_score += 2
        else:
            long_score -= 3  # Strong penalty for counter-trend
        
        # Donchian breakout or continuation
        if donchian_breakout_long:
            long_score += 3
        elif above_donchian:
            long_score += 1
        
        # RSI momentum
        if rsi_strong_bull:
            long_score += 1
        elif rsi_bullish:
            long_score += 0.5
        
        # Regime bonus (trend regime favors breakouts)
        if trending_regime and (donchian_breakout_long or above_donchian):
            long_score += 1
        
        # Enter long if score >= 5 (balanced threshold)
        if long_score >= 5:
            if daily_bullish and hma_bullish:
                new_signal = HIGH_CONV_SIZE
            else:
                new_signal = BASE_SIZE
        
        # SHORT ENTRY
        short_score = 0
        
        # HMA trend (primary)
        if hma_bearish:
            short_score += 2
        
        # Daily trend filter (must align)
        if daily_bearish:
            short_score += 2
        else:
            short_score -= 3  # Strong penalty for counter-trend
        
        # Donchian breakout or continuation
        if donchian_breakout_short:
            short_score += 3
        elif below_donchian:
            short_score += 1
        
        # RSI momentum
        if rsi_strong_bear:
            short_score += 1
        elif rsi_bearish:
            short_score += 0.5
        
        # Regime bonus
        if trending_regime and (donchian_breakout_short or below_donchian):
            short_score += 1
        
        # Enter short if score >= 5
        if short_score >= 5:
            if daily_bearish and hma_bearish:
                new_signal = -HIGH_CONV_SIZE
            else:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # Ensure minimum trade frequency (avoid 0-trade failure)
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            # Relaxed entry after long silence
            if above_donchian and hma_bullish and rsi_bullish:
                new_signal = BASE_SIZE
            elif below_donchian and hma_bearish and rsi_bearish:
                new_signal = -BASE_SIZE
        
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
            # Exit long if HMA turns bearish
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            # Exit short if HMA turns bullish
            if position_side < 0 and hma_bullish:
                trend_reversal = True
        
        # === DONCHIAN MID EXIT ===
        donchian_exit = False
        if in_position and position_side != 0:
            # Exit long if price falls below Donchian mid
            if position_side > 0 and close[i] < donchian_mid[i]:
                donchian_exit = True
            # Exit short if price rises above Donchian mid
            if position_side < 0 and close[i] > donchian_mid[i]:
                donchian_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or donchian_exit:
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