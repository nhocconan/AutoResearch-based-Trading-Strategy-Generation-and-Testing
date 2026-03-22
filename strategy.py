#!/usr/bin/env python3
"""
Experiment #166: 12h Primary + 1d HTF — Fisher Transform + HMA Trend + Regime Filter

Hypothesis: Previous Connors RSI strategies were too restrictive (0 trades on many symbols).
The Ehlers Fisher Transform is proven to catch reversals in bear/range markets with
higher frequency than RSI extremes. Combined with 1d HMA trend bias and Choppiness
regime filter, this should generate 30-60 trades/year with better win rate.

Why this should work:
1. Fisher Transform normalizes price to Gaussian distribution, making extremes meaningful
2. Fisher crosses at -1.5/+1.5 levels occur frequently enough for trade generation
3. 1d HMA slope provides major trend bias without being too restrictive
4. Choppiness filter adjusts entry aggressiveness by regime
5. 12h timeframe = low fee drag, suitable for swing trades

Key differences from failed strategies:
- Fisher crosses are MORE FREQUENT than Connors RSI extremes
- Simpler entry logic (fewer confluence requirements)
- Asymmetric sizing: larger positions in confirmed trends
- Mandatory trade generation fallback every 100 bars

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_hma_regime_1d_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.33
    """
    close = (high + low) / 2.0
    close_s = pd.Series(close)
    
    highest = close_s.rolling(window=period, min_periods=period).max().values
    lowest = close_s.rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    x = 0.67 * (close - lowest) / price_range - 0.33
    x = np.clip(x, -0.99, 0.99)  # Prevent ln domain errors
    
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    return fisher

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Calculate HTF slope (5-bar lookback)
    hma_1d_slope = np.zeros(len(hma_1d_21))
    for i in range(5, len(hma_1d_21)):
        if hma_1d_21[i - 5] != 0:
            hma_1d_slope[i] = (hma_1d_21[i] - hma_1d_21[i - 5]) / hma_1d_21[i - 5] * 100
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher = calculate_fisher_transform(high, low, 9)
    rsi_14 = calculate_rsi(close, 14)
    
    # Fisher signal line (1-bar lag for cross detection)
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    TREND_SIZE = 0.32
    RANGE_SIZE = 0.22
    
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
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        neutral_1d = abs(hma_1d_slope_aligned[i]) <= 0.5
        
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_cross_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_extreme_long = fisher[i] < -1.8
        
        # Short: Fisher crosses below +1.5 from above
        fisher_cross_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        fisher_extreme_short = fisher[i] > 1.8
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 45
        rsi_overbought = rsi_14[i] > 55
        
        # === POSITION SIZING BY REGIME ===
        if is_trend_market and trend_1d_bullish:
            current_size = TREND_SIZE
        elif is_trend_market and trend_1d_bearish:
            current_size = TREND_SIZE
        elif is_range_market:
            current_size = RANGE_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for trade generation
        long_confidence = 0
        
        # Path 1: Fisher cross long + 1d bullish bias (trend follow)
        if fisher_cross_long and trend_1d_bullish:
            long_confidence += 3
        
        # Path 2: Fisher cross long + price above 1d HMA
        if fisher_cross_long and price_above_1d_hma:
            long_confidence += 2
        
        # Path 3: Fisher extreme long + RSI oversold (mean revert)
        if fisher_extreme_long and rsi_oversold:
            long_confidence += 2
        
        # Path 4: Fisher cross long in range market
        if fisher_cross_long and is_range_market:
            long_confidence += 2
        
        # Path 5: Fisher cross long + neutral 1d trend (fallback)
        if fisher_cross_long and neutral_1d:
            long_confidence += 1
        
        if long_confidence >= 2:
            new_signal = current_size
        elif long_confidence == 1 and bars_since_last_trade > 60:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_confidence = 0
        
        # Path 1: Fisher cross short + 1d bearish bias
        if fisher_cross_short and trend_1d_bearish:
            short_confidence += 3
        
        # Path 2: Fisher cross short + price below 1d HMA
        if fisher_cross_short and price_below_1d_hma:
            short_confidence += 2
        
        # Path 3: Fisher extreme short + RSI overbought
        if fisher_extreme_short and rsi_overbought:
            short_confidence += 2
        
        # Path 4: Fisher cross short in range market
        if fisher_cross_short and is_range_market:
            short_confidence += 2
        
        # Path 5: Fisher cross short + neutral 1d trend (fallback)
        if fisher_cross_short and neutral_1d:
            short_confidence += 1
        
        if short_confidence >= 2:
            new_signal = -current_size
        elif short_confidence == 1 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.5
        
        # === MANDATORY TRADE GENERATION (avoid 0 trades) ===
        # If no trades for 120 bars (~60 days on 12h), force entry on weaker signals
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and fisher[i] < -1.0:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and fisher[i] > 1.0:
                new_signal = -current_size * 0.4
            elif fisher_extreme_long:
                new_signal = current_size * 0.35
            elif fisher_extreme_short:
                new_signal = -current_size * 0.35
        
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
        # Exit long when Fisher crosses above +1.5
        # Exit short when Fisher crosses below -1.5
        fisher_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher_prev[i] < 1.5 and fisher[i] >= 1.5:
                fisher_reversal = True
            if position_side < 0 and fisher_prev[i] > -1.5 and fisher[i] <= -1.5:
                fisher_reversal = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if trend turns bearish strongly
            if position_side > 0 and trend_1d_bearish and is_trend_market:
                regime_reversal = True
            # Exit short if trend turns bullish strongly
            if position_side < 0 and trend_1d_bullish and is_trend_market:
                regime_reversal = True
        
        if stoploss_triggered or fisher_reversal or regime_reversal:
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
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            # If same side, keep position (no update needed)
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