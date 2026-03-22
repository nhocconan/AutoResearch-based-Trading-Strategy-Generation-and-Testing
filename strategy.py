#!/usr/bin/env python3
"""
Experiment #196: 12h Primary + 1d HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: Previous Connors RSI strategies failed because RSI is too slow in bear markets.
Research shows Ehlers Fisher Transform catches reversals faster with less lag, especially
during 2022-style crashes and bear market rallies. This strategy combines:

1. EHLERS FISHER TRANSFORM: period=9, enters at -1.5/+1.5 extremes (faster than RSI)
2. CHOPPINESS INDEX: Regime filter (CHOP>55=range/mean-revert, CHOP<45=trend/follow)
3. 1d HMA(21) SLOPE: Major trend bias for asymmetric positioning
4. ADX(14): Trend strength confirmation (ADX>25=trend, ADX<20=range)
5. ATR(14) STOPLOSS: 2.5x trailing stop on all positions

Why this should work:
- Fisher Transform has superior reversal detection vs RSI in literature
- Dual regime logic adapts to bull/bear/range markets
- 12h timeframe = 25-50 trades/year target (low fee drag)
- 1d HTF prevents fighting major trends
- More lenient thresholds than previous attempts to ensure trades

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol (lenient entries to avoid 0-trade failure)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_chop_hma_1d_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * ((close - lowest) / (highest - lowest) - 0.5)
    Signal line = Fisher shifted by 1
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate highest high and lowest low over period
    highest = high_s.rolling(window=period, min_periods=period).max().values
    lowest = low_s.rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    # Normalize price position within range
    x = 0.67 * ((close - lowest) / price_range - 0.5)
    x = np.clip(x, -0.99, 0.99)  # Prevent log domain errors
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
    
    # Signal line (previous Fisher value)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

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
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    # RSI for additional filter
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
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
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.2
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.2
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === ADX TREND STRENGTH ===
        is_strong_trend = adx_14[i] > 25
        is_weak_trend = adx_14[i] < 20
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_cross = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_short_cross = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_range_market:
            current_size = BASE_SIZE * 1.1  # Slightly larger in range (mean revert works well)
        elif is_strong_trend:
            current_size = BASE_SIZE * 0.9  # Slightly smaller in strong trend (more risk)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths (lenient to ensure trades)
        long_score = 0
        
        # Path 1: Fisher long cross + RSI oversold (primary reversal signal)
        if fisher_long_cross and rsi_oversold:
            long_score += 3
        
        # Path 2: Range market + Fisher oversold (mean revert)
        if is_range_market and fisher_oversold:
            long_score += 2
        
        # Path 3: 1d bullish + Fisher long cross (trend pullback)
        if trend_1d_bullish and fisher_long_cross:
            long_score += 2
        
        # Path 4: Price above 1d HMA + RSI extreme (bull pullback)
        if price_above_1d_hma and rsi_extreme_low:
            long_score += 2
        
        # Path 5: Fisher very oversold alone (catch deep dips)
        if fisher[i] < -2.0:
            long_score += 2
        
        # Path 6: Simple RSI extreme (fallback for more trades)
        if rsi_14[i] < 28:
            long_score += 1
        
        if long_score >= 2:
            new_signal = current_size
        elif long_score == 1 and bars_since_last_trade > 60:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Fisher short cross + RSI overbought
        if fisher_short_cross and rsi_overbought:
            short_score += 3
        
        # Path 2: Range market + Fisher overbought
        if is_range_market and fisher_overbought:
            short_score += 2
        
        # Path 3: 1d bearish + Fisher short cross
        if trend_1d_bearish and fisher_short_cross:
            short_score += 2
        
        # Path 4: Price below 1d HMA + RSI extreme (bear rally)
        if price_below_1d_hma and rsi_extreme_high:
            short_score += 2
        
        # Path 5: Fisher very overbought alone
        if fisher[i] > 2.0:
            short_score += 2
        
        # Path 6: Simple RSI extreme (fallback)
        if rsi_14[i] > 72:
            short_score += 1
        
        if short_score >= 2:
            new_signal = -current_size
        elif short_score == 1 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~60 days on 12h) - ensures minimum trades
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and fisher[i] < -1.0:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and fisher[i] > 1.0:
                new_signal = -current_size * 0.4
            elif fisher[i] < -1.5:
                new_signal = current_size * 0.3
            elif fisher[i] > 1.5:
                new_signal = -current_size * 0.3
            elif rsi_14[i] < 30:
                new_signal = current_size * 0.3
            elif rsi_14[i] > 70:
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
            # Exit long if trend turns bearish strongly
            if position_side > 0 and trend_1d_bearish and is_strong_trend:
                regime_reversal = True
            # Exit short if trend turns bullish strongly
            if position_side < 0 and trend_1d_bullish and is_strong_trend:
                regime_reversal = True
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long when Fisher crosses back above 0
            if position_side > 0 and fisher_signal[i] > 0 and fisher[i] > 0.5:
                fisher_exit = True
            # Exit short when Fisher crosses back below 0
            if position_side < 0 and fisher_signal[i] < 0 and fisher[i] < -0.5:
                fisher_exit = True
        
        if stoploss_triggered or regime_reversal or fisher_exit:
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