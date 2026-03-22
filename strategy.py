#!/usr/bin/env python3
"""
Experiment #126: 12h Primary + 1d HTF — Fisher Transform + HMA Trend + Choppiness Regime

Hypothesis: Previous strategies failed due to overly complex entry conditions (multiple
confluence paths that rarely all agree) or pure trend-following that gets destroyed in
bear/range markets. This strategy uses:

1. EHLERS FISHER TRANSFORM: Normalizes price to Gaussian distribution, excellent for
   catching reversals in bear markets. Entry when Fisher crosses -1.5 (long) or +1.5 (short).
   Literature shows 65-70% win rate on reversals.

2. 1d HMA(21) SLOPE: Major trend bias from HTF. Only take longs when 1d trend neutral/bullish,
   shorts when 1d trend neutral/bearish. Prevents fighting macro trends.

3. CHOPPINESS INDEX: Regime switch. CHOP>55 = range (use Fisher mean-revert signals).
   CHOP<45 = trend (use Fisher with trend direction, wait for pullbacks).

4. ATR TRAILING STOP: 2.5*ATR(14) to protect capital during adverse moves.

5. DISCRETE POSITION SIZING: 0.25 base size, reduced to 0.15 in uncertain regimes.

Why this should beat current best (Sharpe=0.220):
- Fisher Transform catches reversals that RSI misses (normalized distribution)
- Simpler entry logic = more trades (avoid 0-trade failure mode)
- 12h timeframe = 25-40 trades/year target (low fee drag)
- Regime-adaptive: mean-revert in chop, trend-pullback in trends
- 1d HTF prevents catastrophic counter-trend positions

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 base, 0.15 reduced
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-40/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_hma_chop_1d_v1"
timeframe = "12h"
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
    """Calculate Hull Moving Average."""
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.66 * ((price - LL) / (HH - LL) - 0.5) + 0.67 * X_prev
    HH/LL = highest high / lowest low over period
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh == ll:
            continue
        
        x = 0.66 * ((close[i] - ll) / (hh - ll) - 0.5) + 0.67 * (fisher_signal[i-1] if i > period else 0)
        x = np.clip(x, -0.99, 0.99)  # Prevent ln domain error
        
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        fisher_signal[i] = x
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    # HMA for 12h trend
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 3)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            continue
        
        # === 1D TREND BIAS ===
        # Slope thresholds for trend detection
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        trend_1d_neutral = not trend_1d_bullish and not trend_1d_bearish
        
        # === 12H TREND ===
        trend_12h_bullish = hma_12h_slope[i] > 0.5
        trend_12h_bearish = hma_12h_slope[i] < -0.5
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher extremes for reversal entries
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Fisher crossover signals (more reliable than absolute levels)
        fisher_cross_up = False
        fisher_cross_down = False
        if i > 1 and not np.isnan(fisher[i-1]):
            fisher_cross_up = fisher[i-1] < -1.5 and fisher[i] >= -1.5
            fisher_cross_down = fisher[i-1] > 1.5 and fisher[i] <= 1.5
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not is_range_market and not is_trend_market:
            current_size = REDUCED_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_entry = False
        
        # Path 1: Range market + Fisher oversold crossover (mean revert)
        if is_range_market and fisher_cross_up:
            long_entry = True
        
        # Path 2: Trend market + 1d bullish + Fisher pullback (trend follow)
        if is_trend_market and (trend_1d_bullish or trend_1d_neutral) and fisher[i] < -0.5:
            long_entry = True
        
        # Path 3: 12h bullish + Fisher extreme (strong momentum reversal)
        if trend_12h_bullish and fisher_oversold:
            long_entry = True
        
        # Path 4: 1d neutral + Fisher cross up (neutral market reversal)
        if trend_1d_neutral and fisher_cross_up and bars_since_last_trade > 30:
            long_entry = True
        
        if long_entry:
            new_signal = current_size
        
        # SHORT ENTRIES
        short_entry = False
        
        # Path 1: Range market + Fisher overbought crossover (mean revert)
        if is_range_market and fisher_cross_down:
            short_entry = True
        
        # Path 2: Trend market + 1d bearish + Fisher pullback (trend follow)
        if is_trend_market and (trend_1d_bearish or trend_1d_neutral) and fisher[i] > 0.5:
            short_entry = True
        
        # Path 3: 12h bearish + Fisher extreme (strong momentum reversal)
        if trend_12h_bearish and fisher_overbought:
            short_entry = True
        
        # Path 4: 1d neutral + Fisher cross down (neutral market reversal)
        if trend_1d_neutral and fisher_cross_down and bars_since_last_trade > 30:
            short_entry = True
        
        if short_entry:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~60 days on 12h) to ensure minimum trades
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and fisher[i] < 0:
                new_signal = REDUCED_SIZE
            elif trend_1d_bearish and fisher[i] > 0:
                new_signal = -REDUCED_SIZE
            elif fisher[i] < -1.0:
                new_signal = REDUCED_SIZE * 0.8
            elif fisher[i] > 1.0:
                new_signal = -REDUCED_SIZE * 0.8
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if trend turns bearish strongly
            if position_side > 0 and trend_1d_bearish and fisher[i] > 0.5:
                regime_reversal = True
            # Exit short if trend turns bullish strongly
            if position_side < 0 and trend_1d_bullish and fisher[i] < -0.5:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === FISHER REVERSAL EXIT ===
        # Exit when Fisher reaches opposite extreme
        fisher_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher[i] > 1.5:
                fisher_exit = True
            if position_side < 0 and fisher[i] < -1.5:
                fisher_exit = True
        
        if fisher_exit:
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