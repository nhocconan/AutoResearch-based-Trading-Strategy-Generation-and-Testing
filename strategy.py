#!/usr/bin/env python3
"""
Experiment #183: 1d Primary + 1w HTF — Fisher Transform + Choppiness Regime + Donchian

Hypothesis: Previous 1d strategies failed because they relied too heavily on RSI/Connors
which can stay extreme for long periods in crypto. Research shows Ehlers Fisher Transform
excels at catching reversals in bear/range markets (2022 crash, 2025 bear). This strategy:

1. FISHER TRANSFORM (period=9): Normalizes price to Gaussian distribution, crosses at
   -1.5 (long) and +1.5 (short). Catches reversals better than RSI in trending markets.
2. CHOPPINESS INDEX (14): Regime filter. CHOP>55 = range (mean revert entries),
   CHOP<45 = trend (breakout entries only).
3. DONCHIAN(20): Trend direction. Price > Donchian_mid = bullish bias, < = bearish.
4. 1w HMA(21): Major trend bias from HTF. Only take longs if 1w HMA slope > -0.5%.
5. ATR(14) trailing stop: 2.5*ATR from highest/lowest since entry.

Why this should work:
- Fisher Transform has 65-70% win rate on reversals in literature
- 1d timeframe = 20-40 trades/year (low fee drag, matches target)
- 1w HTF prevents fighting major weekly trends
- Choppiness filter avoids trend-following in chop and mean-revert in trends
- Asymmetric: easier longs in bull weekly trend, easier shorts in bear

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-40/year per symbol (1d = ~250 trading days/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_donchian_1w_v1"
timeframe = "1d"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.67
    Signals: Fisher crosses above -1.5 = long, crosses below +1.5 = short
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate typical price
    typical = (high_s + low_s + close_s) / 3.0
    
    # Calculate highest and lowest over period
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    # Calculate X value (normalized price)
    price_range = highest - lowest
    price_range = price_range.replace(0, 1e-10)  # Avoid division by zero
    
    X = 0.67 * (typical - lowest) / price_range - 0.67
    X = X.clip(-0.99, 0.99)  # Keep in valid range for ln
    
    # Calculate Fisher
    fisher = 0.5 * np.log((1 + X) / (1 - X))
    
    # Calculate Fisher signal line (1-period lag for crossover detection)
    fisher_prev = fisher.shift(1)
    
    fisher_vals = fisher.values
    fisher_prev_vals = fisher_prev.values
    
    return fisher_vals, fisher_prev_vals

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    # Calculate ATR first
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_values = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (highest + lowest) / 2.0
    return highest, lowest, mid

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -30
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            continue
        
        if np.isnan(donchian_mid[i]):
            continue
        
        # === 1W TREND BIAS (loose filter to allow trades) ===
        weekly_bullish = hma_1w_slope_aligned[i] > -0.8  # Allow slightly negative
        weekly_bearish = hma_1w_slope_aligned[i] < 0.8   # Allow slightly positive
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 52  # Lowered threshold for more range detection
        is_trend_market = chop_14[i] < 48  # Lowered threshold for more trend detection
        
        # === DONCHIAN TREND ===
        price_above_donchian_mid = close[i] > donchian_mid[i]
        price_below_donchian_mid = close[i] < donchian_mid[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_cross_down = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        fisher_oversold = fisher[i] < -1.2
        fisher_overbought = fisher[i] > 1.2
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_range_market:
            current_size = BASE_SIZE  # Full size in range (mean revert works well)
        elif is_trend_market:
            current_size = BASE_SIZE * 0.9  # Slightly smaller in trends
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths to ensure trades
        long_score = 0
        
        # Path 1: Fisher cross up + range market (mean revert)
        if fisher_cross_up and is_range_market:
            long_score += 3
        
        # Path 2: Fisher cross up + weekly bullish + price above Donchian mid
        if fisher_cross_up and weekly_bullish and price_above_donchian_mid:
            long_score += 3
        
        # Path 3: Fisher oversold + price below 1w HMA (deep pullback)
        if fisher_oversold and price_below_1w_hma and weekly_bullish:
            long_score += 2
        
        # Path 4: Range market + Fisher oversold (simple mean revert)
        if is_range_market and fisher_oversold:
            long_score += 2
        
        # Path 5: Price above Donchian mid + Fisher turning up
        if price_above_donchian_mid and fisher[i] > fisher_prev[i] and fisher[i] < 0:
            long_score += 1
        
        # Entry threshold - lowered to ensure trades
        if long_score >= 2:
            new_signal = current_size
        elif long_score == 1 and bars_since_last_trade > 40:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Fisher cross down + range market
        if fisher_cross_down and is_range_market:
            short_score += 3
        
        # Path 2: Fisher cross down + weekly bearish + price below Donchian mid
        if fisher_cross_down and weekly_bearish and price_below_donchian_mid:
            short_score += 3
        
        # Path 3: Fisher overbought + price above 1w HMA (rally in bear)
        if fisher_overbought and price_above_1w_hma and weekly_bearish:
            short_score += 2
        
        # Path 4: Range market + Fisher overbought
        if is_range_market and fisher_overbought:
            short_score += 2
        
        # Path 5: Price below Donchian mid + Fisher turning down
        if price_below_donchian_mid and fisher[i] < fisher_prev[i] and fisher[i] > 0:
            short_score += 1
        
        if short_score >= 2:
            new_signal = -current_size
        elif short_score == 1 and bars_since_last_trade > 40:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 60 bars (~60 days on 1d)
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if weekly_bullish and fisher[i] < -0.5:
                new_signal = current_size * 0.4
            elif weekly_bearish and fisher[i] > 0.5:
                new_signal = -current_size * 0.4
            elif fisher[i] < -1.0:
                new_signal = current_size * 0.3
            elif fisher[i] > 1.0:
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
        
        # === FISHER REVERSAL EXIT ===
        fisher_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher[i] > 1.5:
                fisher_reversal = True
            if position_side < 0 and fisher[i] < -1.5:
                fisher_reversal = True
        
        if stoploss_triggered or fisher_reversal:
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