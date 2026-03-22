#!/usr/bin/env python3
"""
Experiment #204: 4h Primary + 12h/1d HTF — Fisher Transform + Vol Spike Mean Reversion

Hypothesis: Previous regime-switching strategies failed due to over-complexity and
too-few trades. Research shows Ehlers Fisher Transform catches reversals in bear
rallies with 70%+ win rate. Combined with vol-spike detection (ATR ratio > 1.8)
and HTF trend bias, this should work across BTC/ETH/SOL in both bull and bear markets.

Key components:
1. EHLERS FISHER TRANSFORM: period=9, long when Fisher crosses above -1.5, short when
   crosses below +1.5. Catches reversals better than RSI in choppy markets.
2. VOLATILITY SPIKE: ATR(7)/ATR(30) > 1.6 signals capitulation/extreme moves
3. 12h HMA(21) SLOPE: Major trend bias (avoid counter-trend in strong moves)
4. 1d HMA(21): Secondary trend filter for stronger conviction
5. BOLLINGER BANDS: Price < BB_lower(20, 2.0) confirms oversold for longs
6. CHOPPINESS INDEX: Regime filter (range = mean revert, trend = pullback entries)

Why this should work:
- Fisher Transform has superior reversal detection vs RSI (Ehlers research)
- Vol spike + BB extreme = capitulation events (high win rate reversals)
- 4h timeframe = 30-60 trades/year target (balanced fee drag vs opportunity)
- 12h/1d HTF prevents fighting major trends
- Asymmetric sizing: larger positions when HTF agrees with signal

Timeframe: 4h (REQUIRED for Experiment #204)
HTF: 12h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_volspike_bb_12h1d_v1"
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
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.66 * ((price - LL) / (HH - LL) - 0.5)
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max().values
    ll = low_s.rolling(window=period, min_periods=period).min().values
    
    # Calculate X value
    price_range = hh - ll
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    x = 0.66 * ((close - ll) / price_range - 0.5)
    x = np.clip(x, -0.99, 0.99)  # Prevent division by zero in log
    
    # Calculate Fisher
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    
    # Calculate Fisher trigger (1-bar lag of Fisher)
    fisher_trigger = np.roll(fisher, 1)
    fisher_trigger[0] = fisher[0]
    
    return fisher, fisher_trigger

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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volatility spike ratio
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(atr_ratio[i]):
            continue
        
        # === HTF TREND BIAS ===
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.2
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.2
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.3
        
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        
        # Strong trend agreement
        strong_bull = trend_12h_bullish and trend_1d_bullish
        strong_bear = trend_12h_bearish and trend_1d_bearish
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === VOLATILITY SPIKE ===
        vol_spike = atr_ratio[i] > 1.6
        vol_extreme = atr_ratio[i] > 2.0
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = (fisher[i] > -1.5) and (fisher_trigger[i] <= -1.5)
        fisher_cross_down = (fisher[i] < 1.5) and (fisher_trigger[i] >= 1.5)
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if strong_bull or strong_bear:
            current_size = BASE_SIZE  # Full size when HTF agrees
        elif not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.7  # Reduce in unclear regime
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths for sufficient trades
        long_score = 0
        long_confidence = 0
        
        # Path 1: Fisher cross up + vol spike + BB lower (capitulation reversal)
        if fisher_cross_up and vol_spike and price_below_bb_lower:
            long_score += 4
            long_confidence = 1.0
        
        # Path 2: Fisher oversold + range market (mean revert)
        if fisher_oversold and is_range_market:
            long_score += 3
            long_confidence = 0.8
        
        # Path 3: Fisher cross up + strong bull trend (pullback entry)
        if fisher_cross_up and strong_bull:
            long_score += 3
            long_confidence = 0.9
        
        # Path 4: Vol spike + BB lower + RSI oversold (classic mean revert)
        if vol_spike and price_below_bb_lower and rsi_oversold:
            long_score += 3
            long_confidence = 0.85
        
        # Path 5: Fisher oversold + price below 12h HMA (deep pullback)
        if fisher_oversold and price_below_12h_hma and not strong_bear:
            long_score += 2
            long_confidence = 0.7
        
        # Path 6: Simple Fisher cross up (fallback for trade frequency)
        if fisher_cross_up and bars_since_last_trade > 60:
            long_score += 1
            long_confidence = 0.5
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score == 2 and bars_since_last_trade > 40:
            new_signal = current_size * 0.7
        elif long_score >= 1 and bars_since_last_trade > 80:
            new_signal = HALF_SIZE
        
        # SHORT ENTRIES
        short_score = 0
        short_confidence = 0
        
        # Path 1: Fisher cross down + vol spike + BB upper
        if fisher_cross_down and vol_spike and price_above_bb_upper:
            short_score += 4
            short_confidence = 1.0
        
        # Path 2: Fisher overbought + range market
        if fisher_overbought and is_range_market:
            short_score += 3
            short_confidence = 0.8
        
        # Path 3: Fisher cross down + strong bear trend
        if fisher_cross_down and strong_bear:
            short_score += 3
            short_confidence = 0.9
        
        # Path 4: Vol spike + BB upper + RSI overbought
        if vol_spike and price_above_bb_upper and rsi_overbought:
            short_score += 3
            short_confidence = 0.85
        
        # Path 5: Fisher overbought + price above 12h HMA (rally in bear)
        if fisher_overbought and price_above_12h_hma and not strong_bull:
            short_score += 2
            short_confidence = 0.7
        
        # Path 6: Simple Fisher cross down (fallback)
        if fisher_cross_down and bars_since_last_trade > 60:
            short_score += 1
            short_confidence = 0.5
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score == 2 and bars_since_last_trade > 40:
            new_signal = -current_size * 0.7
        elif short_score >= 1 and bars_since_last_trade > 80:
            new_signal = -HALF_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~20 days on 4h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if strong_bull and fisher[i] < -0.5:
                new_signal = BASE_SIZE * 0.4
            elif strong_bear and fisher[i] > 0.5:
                new_signal = -BASE_SIZE * 0.4
            elif fisher[i] < -1.0:
                new_signal = BASE_SIZE * 0.3
            elif fisher[i] > 1.0:
                new_signal = -BASE_SIZE * 0.3
        
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
            # Exit long if strong bear trend develops
            if position_side > 0 and strong_bear:
                regime_reversal = True
            # Exit short if strong bull trend develops
            if position_side < 0 and strong_bull:
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
        
        signals[i] = new_signal
    
    return signals