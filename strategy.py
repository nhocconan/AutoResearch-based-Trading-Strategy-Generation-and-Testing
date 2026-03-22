#!/usr/bin/env python3
"""
Experiment #084: 4h Primary + 12h/1d HTF — Choppiness Regime + Fisher Transform

Hypothesis: Previous 4h strategies failed because they used laggy trend indicators (EMA/HMA crossover)
that whipsaw in range markets. This strategy uses:
1. CHOPPINESS INDEX (14) for regime detection - purpose-built for range vs trend
2. EHLERS FISHER TRANSFORM (9) for entry timing - catches reversals with less lag
3. 12h/1d HMA slope for major trend bias - prevents counter-trend trades
4. Asymmetric logic - different rules for bull/bear/range regimes

Why this should work:
- Fisher Transform normalizes price to Gaussian distribution, extreme values (-2/+2) mark reversals
- Choppiness Index cleanly separates range (CHOP>61.8) from trend (CHOP<38.2) markets
- 12h/1d HTF prevents taking shorts in strong bull markets and vice versa
- 4h timeframe naturally limits trades to 30-60/year (fee-efficient)
- Asymmetric sizing: larger positions in confirmed trends, smaller in ranges

Timeframe: 4h (REQUIRED for exp #084)
HTF: 12h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_regime_12h1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution, extreme values mark reversals.
    Fisher > 1.5 = overbought (potential short)
    Fisher < -1.5 = oversold (potential long)
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate typical price
    typical = (high_s + low_s + close_s) / 3.0
    
    # Normalize price to -1 to +1 range
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    price_range = highest - lowest
    price_range = price_range.replace(0, 1e-10)
    
    normalized = 2 * (typical - lowest) / price_range - 1
    
    # Apply Fisher Transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    fisher = fisher.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period as percentage."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # Band width as percentage
    bw = (upper - lower) / sma * 100
    
    return upper.values, lower.values, bw.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    
    # 4h trend indicators
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
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
        
        if np.isnan(hma_12h_slope_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # Strong bullish: slope > 1.0%
        # Strong bearish: slope < -1.0%
        # Neutral: between
        trend_1d_strong_bull = hma_1d_slope_aligned[i] > 1.0
        trend_1d_strong_bear = hma_1d_slope_aligned[i] < -1.0
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.3
        
        # Price vs 1d HMA
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 12H TREND CONFIRMATION ===
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.5
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.5
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 61.8 = range market (mean revert)
        # CHOP < 38.2 = trend market (trend follow)
        # CHOP between = transitional
        is_range_market = chop_14[i] > 58
        is_trend_market = chop_14[i] < 42
        
        # === 4H TREND ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crossing above -1.5 from below = long signal
        # Fisher crossing below +1.5 from above = short signal
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher_signal[i] < -1.0
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher_signal[i] > 1.0
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === BOLLINGER BAND POSITION ===
        price_near_bb_lower = close[i] < bb_lower[i] * 1.005
        price_near_bb_upper = close[i] > bb_upper[i] * 0.995
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in transitional markets
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.6
        
        # Increase size in strong confirmed trends
        if is_trend_market and trend_1d_strong_bull and trend_12h_bullish:
            current_size = BASE_SIZE * 1.1
        if is_trend_market and trend_1d_strong_bear and trend_12h_bearish:
            current_size = BASE_SIZE * 1.1
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if is_range_market:
            # Mean reversion: buy at BB lower + Fisher oversold + RSI confirmation
            if price_near_bb_lower and fisher_oversold and rsi_oversold:
                # Only if not strongly bearish on 1d
                if not trend_1d_strong_bear:
                    new_signal = current_size
            # Fisher cross up from oversold
            elif fisher_cross_up and rsi_oversold:
                if not trend_1d_strong_bear:
                    new_signal = current_size * 0.8
        
        elif is_trend_market:
            # Trend following: pullback entries in uptrend
            if trend_1d_bullish and trend_12h_bullish and hma_bullish:
                # Buy Fisher cross up from moderate oversold
                if fisher[i] < -0.5 and fisher_cross_up:
                    new_signal = current_size
                # Or buy RSI pullback
                elif rsi_14[i] < 45 and price_above_12h_hma:
                    new_signal = current_size * 0.8
            # Also allow if 1d bullish even without 12h confirmation
            elif trend_1d_strong_bull and fisher[i] < -1.0:
                new_signal = current_size * 0.7
        
        else:
            # Transitional: weaker signals with strong 1d bias
            if trend_1d_bullish and fisher[i] < -1.0 and rsi_oversold:
                new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        if is_range_market:
            # Mean reversion: sell at BB upper + Fisher overbought + RSI confirmation
            if price_near_bb_upper and fisher_overbought and rsi_overbought:
                # Only if not strongly bullish on 1d
                if not trend_1d_strong_bull:
                    new_signal = -current_size
            # Fisher cross down from overbought
            elif fisher_cross_down and rsi_overbought:
                if not trend_1d_strong_bull:
                    new_signal = -current_size * 0.8
        
        elif is_trend_market:
            # Trend following: pullback entries in downtrend
            if trend_1d_bearish and trend_12h_bearish and hma_bearish:
                # Sell Fisher cross down from moderate overbought
                if fisher[i] > 0.5 and fisher_cross_down:
                    new_signal = -current_size
                # Or sell RSI pullback
                elif rsi_14[i] > 55 and price_below_12h_hma:
                    new_signal = -current_size * 0.8
            # Also allow if 1d bearish even without 12h confirmation
            elif trend_1d_strong_bear and fisher[i] > 1.0:
                new_signal = -current_size * 0.7
        
        else:
            # Transitional: weaker signals with strong 1d bias
            if trend_1d_bearish and fisher[i] > 1.0 and rsi_overbought:
                new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~13 days on 4h), allow weaker entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and fisher[i] < -0.5:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and fisher[i] > 0.5:
                new_signal = -current_size * 0.4
        
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
        # Exit if regime changes strongly against position
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if market becomes strongly trending bearish on 1d
            if position_side > 0 and trend_1d_strong_bear:
                regime_reversal = True
            # Exit short if market becomes strongly trending bullish on 1d
            if position_side < 0 and trend_1d_strong_bull:
                regime_reversal = True
        
        # Apply stoploss or regime reversal
        if stoploss_triggered or regime_reversal:
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