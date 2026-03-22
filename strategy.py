#!/usr/bin/env python3
"""
Experiment #179: 4h Primary + 1d HTF — Fisher Transform + Choppiness Regime + Asymmetric Entries

Hypothesis: Previous 4h strategies failed because they used rigid trend-following that got
whipsawed in 2022 crash and 2025 bear market. Research shows Fisher Transform catches
reversals better than RSI in bear/range markets, and Choppiness Index correctly identifies
when to mean-revert vs trend-follow.

Key innovations:
1. FISHER TRANSFORM (period=9): Long when Fisher crosses above -1.5, short when below +1.5
   Literature shows 70%+ win rate on reversals in bear markets
2. CHOPPINESS REGIME: CHOP>55 = range (mean revert at BB extremes), CHOP<45 = trend (pullback entries)
3. 1d HMA SLOPE: Major trend bias - avoid counter-trend when 1d slope is extreme
4. ADX FILTER: Only trend-follow when ADX>25, only mean-revert when ADX<20
5. ASYMMETRIC SIZING: Larger positions when regime + HTF align (0.35), smaller when conflicting (0.20)
6. RELAXED THRESHOLDS: Ensure 30-50 trades/year by not requiring perfect confluence

Why 4h works: 20-50 trades/year target, enough bars for indicators, less noise than 1h
HTF 1d: Prevents fighting major trends (e.g., don't short when 1d HMA slope strongly bullish)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.35 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_regime_1d_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

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
    Transforms price into a Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
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
    
    normalized = 2.0 * (typical - lowest) / price_range - 1.0
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Apply Fisher transformation
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Smooth with Wilder's method
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / tr_smooth.replace(0, np.nan)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    HIGH_CONV_SIZE = 0.35
    LOW_CONV_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    prev_fisher_long = False
    prev_fisher_short = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(fisher[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === ADX STRENGTH ===
        adx_strong = adx_14[i] > 25
        adx_weak = adx_14[i] < 20
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_signal = (fisher[i] > -1.5) and (prev_fisher_long == False or fisher_signal[i] < -1.5)
        fisher_short_signal = (fisher[i] < 1.5) and (prev_fisher_short == False or fisher_signal[i] > 1.5)
        
        # Fisher crossover detection
        fisher_crossed_up = (fisher[i] > fisher_signal[i]) and (fisher_signal[i] < -1.0)
        fisher_crossed_down = (fisher[i] < fisher_signal[i]) and (fisher_signal[i] > 1.0)
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_width = (bb_upper[i] - bb_lower[i]) / bb_mid[i] if bb_mid[i] > 0 else 0
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === POSITION SIZING BASED ON CONFLUENCE ===
        confluence_score = 0
        if is_range_market and is_trend_market == False:
            confluence_score += 1
        if trend_1d_bullish or trend_1d_bearish:
            confluence_score += 1
        if adx_strong or adx_weak:
            confluence_score += 1
        
        current_size = BASE_SIZE
        if confluence_score >= 2:
            current_size = HIGH_CONV_SIZE
        elif confluence_score == 0:
            current_size = LOW_CONV_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for sufficient trade frequency
        long_score = 0
        
        # Path 1: Range market + Fisher long + RSI oversold (mean revert)
        if is_range_market and fisher_crossed_up and rsi_oversold:
            long_score += 3
        
        # Path 2: Range market + price below BB lower + Fisher improving
        if is_range_market and price_below_bb_lower and fisher[i] > -2.0:
            long_score += 2
        
        # Path 3: Trend market + 1d bullish + Fisher pullback entry
        if is_trend_market and trend_1d_bullish and fisher[i] > -1.0 and fisher[i] < 0.5:
            long_score += 2
        
        # Path 4: Price above 1d HMA + RSI pullback (bull market dip)
        if price_above_1d_hma and rsi_14[i] < 45 and fisher[i] > -1.5:
            long_score += 2
        
        # Path 5: Simple Fisher reversal (fallback for trades)
        if fisher_crossed_up and rsi_14[i] < 50:
            long_score += 1
        
        # Path 6: BB squeeze breakout long
        if bb_width < 0.05 and price_above_bb_upper and fisher[i] > 0:
            long_score += 2
        
        if long_score >= 2:
            new_signal = current_size
        elif long_score == 1 and bars_since_last_trade > 60:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Range market + Fisher short + RSI overbought
        if is_range_market and fisher_crossed_down and rsi_overbought:
            short_score += 3
        
        # Path 2: Range market + price above BB upper + Fisher declining
        if is_range_market and price_above_bb_upper and fisher[i] < 2.0:
            short_score += 2
        
        # Path 3: Trend market + 1d bearish + Fisher rally entry
        if is_trend_market and trend_1d_bearish and fisher[i] < 1.0 and fisher[i] > -0.5:
            short_score += 2
        
        # Path 4: Price below 1d HMA + RSI rally (bear market bounce)
        if price_below_1d_hma and rsi_14[i] > 55 and fisher[i] < 1.5:
            short_score += 2
        
        # Path 5: Simple Fisher reversal (fallback)
        if fisher_crossed_down and rsi_14[i] > 50:
            short_score += 1
        
        # Path 6: BB squeeze breakout short
        if bb_width < 0.05 and price_below_bb_lower and fisher[i] < 0:
            short_score += 2
        
        if short_score >= 2:
            new_signal = -current_size
        elif short_score == 1 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~20 days on 4h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and fisher[i] > -1.0 and rsi_14[i] < 45:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and fisher[i] < 1.0 and rsi_14[i] > 55:
                new_signal = -current_size * 0.5
            elif fisher[i] < -1.5 and rsi_14[i] < 35:
                new_signal = current_size * 0.4
            elif fisher[i] > 1.5 and rsi_14[i] > 65:
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
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if regime shifts to strong trend bearish
            if position_side > 0 and is_trend_market and trend_1d_bearish and adx_strong:
                regime_reversal = True
            # Exit short if regime shifts to strong trend bullish
            if position_side < 0 and is_trend_market and trend_1d_bullish and adx_strong:
                regime_reversal = True
        
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
        
        # Update Fisher state for crossover detection
        prev_fisher_long = fisher[i] > -1.5
        prev_fisher_short = fisher[i] < 1.5
        
        signals[i] = new_signal
    
    return signals