#!/usr/bin/env python3
"""
Experiment #116: 12h Primary + 1d HTF — Regime-Adaptive HMA/Donchian Hybrid

Hypothesis: Previous strategies failed due to either (a) too many conflicting filters 
resulting in 0 trades, or (b) pure mean-reversion that fails in strong trends. 
This strategy uses a regime-adaptive approach:

1. REGIME DETECTION: Choppiness Index determines if market is trending (CHOP<45) 
   or ranging (CHOP>55). Different entry logic for each regime.
2. TRENDING REGIME: HMA(21) crossover + Donchian(20) breakout + 1d HMA slope bias
3. RANGING REGIME: RSI(14) extremes + Bollinger Band mean reversion
4. 1d HTF BIAS: Only take trades aligned with daily trend (prevents counter-trend)
5. ATR TRAILING STOP: 2.5x ATR(14) to protect capital

Why this should work:
- Regime switching adapts to market conditions (trend vs range)
- 12h timeframe = 25-50 trades/year target (low fee drag)
- Less strict than Connors RSI (which caused 0 trades in some experiments)
- 1d HTF prevents fighting major trends
- Discrete position sizing (0.30) minimizes fee churn

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_hma_donchian_1d_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

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
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_vals = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

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
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_48 = calculate_hma(close, 48)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, 20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    
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
        
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.2
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.2
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        is_neutral = not is_range_market and not is_trend_market
        
        # === 12h HMA CROSSOVER ===
        hma_bullish_cross = hma_12h_21[i] > hma_12h_48[i]
        hma_bearish_cross = hma_12h_21[i] < hma_12h_48[i]
        
        # === DONCHIAN BREAKOUT ===
        donch_breakout_long = close[i] > donch_upper[i] * 0.998  # Near breakout
        donch_breakout_short = close[i] < donch_lower[i] * 1.002  # Near breakout
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === BOLLINGER POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_neutral:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for sufficient trades
        long_confidence = 0
        
        # Path 1: Trending regime + HMA bullish + 1d bullish bias
        if is_trend_market and hma_bullish_cross and trend_1d_bullish:
            long_confidence += 3
        
        # Path 2: Trending regime + Donchian breakout + 1d bullish
        if is_trend_market and donch_breakout_long and (trend_1d_bullish or price_above_1d_hma):
            long_confidence += 3
        
        # Path 3: Ranging regime + RSI oversold + BB lower (mean revert)
        if is_range_market and rsi_oversold and price_below_bb_lower:
            long_confidence += 2
        
        # Path 4: Ranging regime + RSI extreme low
        if is_range_market and rsi_extreme_low:
            long_confidence += 2
        
        # Path 5: Neutral regime + HMA bullish + RSI not overbought
        if is_neutral and hma_bullish_cross and rsi_14[i] < 60:
            long_confidence += 2
        
        # Path 6: 1d bullish + pullback to HMA (any regime)
        if trend_1d_bullish and close[i] < hma_12h_21[i] * 1.02 and rsi_oversold:
            long_confidence += 2
        
        # Path 7: Simple fallback for trade generation
        if hma_bullish_cross and rsi_14[i] < 50 and bars_since_last_trade > 60:
            long_confidence += 1
        
        if long_confidence >= 3:
            new_signal = current_size
        elif long_confidence == 2 and bars_since_last_trade > 40:
            new_signal = current_size
        elif long_confidence == 1 and bars_since_last_trade > 80:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_confidence = 0
        
        # Path 1: Trending regime + HMA bearish + 1d bearish bias
        if is_trend_market and hma_bearish_cross and trend_1d_bearish:
            short_confidence += 3
        
        # Path 2: Trending regime + Donchian breakdown + 1d bearish
        if is_trend_market and donch_breakout_short and (trend_1d_bearish or price_below_1d_hma):
            short_confidence += 3
        
        # Path 3: Ranging regime + RSI overbought + BB upper (mean revert)
        if is_range_market and rsi_overbought and price_above_bb_upper:
            short_confidence += 2
        
        # Path 4: Ranging regime + RSI extreme high
        if is_range_market and rsi_extreme_high:
            short_confidence += 2
        
        # Path 5: Neutral regime + HMA bearish + RSI not oversold
        if is_neutral and hma_bearish_cross and rsi_14[i] > 40:
            short_confidence += 2
        
        # Path 6: 1d bearish + rally to HMA (any regime)
        if trend_1d_bearish and close[i] > hma_12h_21[i] * 0.98 and rsi_overbought:
            short_confidence += 2
        
        # Path 7: Simple fallback for trade generation
        if hma_bearish_cross and rsi_14[i] > 50 and bars_since_last_trade > 60:
            short_confidence += 1
        
        if short_confidence >= 3:
            new_signal = -current_size
        elif short_confidence == 2 and bars_since_last_trade > 40:
            new_signal = -current_size
        elif short_confidence == 1 and bars_since_last_trade > 80:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~60 days on 12h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.4
            elif hma_bullish_cross and rsi_14[i] < 50:
                new_signal = current_size * 0.3
            elif hma_bearish_cross and rsi_14[i] > 50:
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if regime shifts to strong bearish trend
            if position_side > 0 and is_trend_market and trend_1d_bearish and hma_bearish_cross:
                regime_reversal = True
            # Exit short if regime shifts to strong bullish trend
            if position_side < 0 and is_trend_market and trend_1d_bullish and hma_bullish_cross:
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