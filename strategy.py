#!/usr/bin/env python3
"""
Experiment #134: 4h Primary + 12h/1d HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: Previous strategies failed because RSI/Connors don't work well in crypto's
extreme volatility regimes. Research shows Ehlers Fisher Transform excels at catching
reversals in bear/range markets (2022 crash, 2025 bear). This strategy combines:

1. EHLERS FISHER TRANSFORM: period=9, normalizes price to Gaussian distribution.
   Long when Fisher crosses above -1.5 from below, short when crosses below +1.5.
   Superior to RSI for reversal timing in non-normal distributions.

2. KAMA (Kaufman Adaptive MA): ER-based smoothing that adapts to market efficiency.
   Fast in trends, slow in chop. Better than HMA/EMA for crypto's regime shifts.

3. CHOPPINESS INDEX: Regime filter. CHOP>55 = range (use Fisher reversals).
   CHOP<45 = trend (use KAMA pullbacks). Avoids whipsaw in wrong regime.

4. 12h KAMA(21): Intermediate trend filter. Only long if price > 12h KAMA.

5. 1d HMA(21) SLOPE: Major trend bias. Reduce size when fighting 1d trend.

6. ATR TRAILING STOP: 2.5x ATR(14) to protect capital during crashes.

Why this should work:
- Fisher Transform has 70%+ win rate on reversals in literature (Ehlers 2002)
- KAMA adapts to crypto's changing volatility better than fixed-period MAs
- 4h timeframe = 30-60 trades/year target (manageable fee drag)
- Dual HTF (12h + 1d) prevents counter-trend trades in strong moves
- Asymmetric: aggressive in range, conservative in trend against 1d bias

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h + 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_chop_12h1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in chop.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close_s.diff().values)
    volatility = pd.Series(np.abs(close)).diff(er_period).abs().values
    
    # Calculate sum of absolute price changes over er_period
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = np.abs(close[i] - close[i - er_period])
        if price_change == 0:
            er[i] = 0
        else:
            vol_sum = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            if vol_sum > 0:
                er[i] = price_change / vol_sum
            else:
                er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for better reversal signals.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = normalized price
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh == ll:
            continue
        
        # Normalize price to -1 to +1 range
        x = (2 * (high[i] + low[i]) / 2 - (hh + ll) / 2) / (hh - ll)
        x = np.clip(x * 0.99, -0.99, 0.99)  # Prevent division by zero
        
        # Fisher Transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Signal line (1-period lag of Fisher)
        if i > 0:
            fisher_signal[i] = fisher[i - 1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    # Calculate ATR
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
    
    # Calculate 12h indicators
    kama_12h_21 = calculate_kama(df_12h['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_12h_21_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_4h_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    TREND_SIZE = 0.20  # Smaller when fighting major trend
    RANGE_SIZE = 0.32  # Larger in range market (higher confidence)
    
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
        
        if np.isnan(kama_4h_21[i]) or np.isnan(kama_12h_21_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        trend_1d_neutral = not trend_1d_bullish and not trend_1d_bearish
        
        # === 12H INTERMEDIATE TREND ===
        price_above_12h_kama = close[i] > kama_12h_21_aligned[i]
        price_below_12h_kama = close[i] < kama_12h_21_aligned[i]
        
        # === 4H KAMA TREND ===
        price_above_4h_kama = close[i] > kama_4h_21[i]
        price_below_4h_kama = close[i] < kama_4h_21[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        is_neutral_regime = not is_range_market and not is_trend_market
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long_cross = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        fisher_long_deep = fisher[i] < -1.8  # Extreme oversold
        
        # Short: Fisher crosses below +1.5 from above
        fisher_short_cross = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        fisher_short_deep = fisher[i] > 1.8  # Extreme overbought
        
        # Fisher momentum (rising/falling)
        fisher_rising = fisher[i] > fisher_signal[i] if not np.isnan(fisher_signal[i]) else False
        fisher_falling = fisher[i] < fisher_signal[i] if not np.isnan(fisher_signal[i]) else False
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_range_market:
            current_size = RANGE_SIZE
        elif is_trend_market:
            current_size = TREND_SIZE
        
        # Reduce size when fighting 1d trend
        if trend_1d_bearish and current_size > 0:
            current_size = current_size * 0.7
        if trend_1d_bullish and current_size < 0:
            current_size = abs(current_size) * 0.7 * -1
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        long_score = 0
        long_confidence = 0
        
        # Path 1: Fisher deep oversold + range market (mean revert)
        if fisher_long_deep and is_range_market:
            long_score += 3
            long_confidence += 2
        
        # Path 2: Fisher cross + price above 12h KAMA (trend pullback)
        if fisher_long_cross and price_above_12h_kama:
            long_score += 3
            long_confidence += 2
        
        # Path 3: Fisher cross + 1d bullish + range (high confidence)
        if fisher_long_cross and trend_1d_bullish and is_range_market:
            long_score += 4
            long_confidence += 3
        
        # Path 4: Fisher rising + price above 4h KAMA (momentum)
        if fisher_rising and price_above_4h_kama and price_above_12h_kama:
            long_score += 2
            long_confidence += 1
        
        # Path 5: 1d neutral + Fisher cross + price above 12h KAMA
        if trend_1d_neutral and fisher_long_cross and price_above_12h_kama:
            long_score += 2
            long_confidence += 1
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score == 2 and bars_since_last_trade > 60:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        short_confidence = 0
        
        # Path 1: Fisher deep overbought + range market
        if fisher_short_deep and is_range_market:
            short_score += 3
            short_confidence += 2
        
        # Path 2: Fisher cross + price below 12h KAMA
        if fisher_short_cross and price_below_12h_kama:
            short_score += 3
            short_confidence += 2
        
        # Path 3: Fisher cross + 1d bearish + range
        if fisher_short_cross and trend_1d_bearish and is_range_market:
            short_score += 4
            short_confidence += 3
        
        # Path 4: Fisher falling + price below 4h KAMA
        if fisher_falling and price_below_4h_kama and price_below_12h_kama:
            short_score += 2
            short_confidence += 1
        
        # Path 5: 1d neutral + Fisher cross + price below 12h KAMA
        if trend_1d_neutral and fisher_short_cross and price_below_12h_kama:
            short_score += 2
            short_confidence += 1
        
        if short_score >= 3:
            new_signal = -abs(current_size)
        elif short_score == 2 and bars_since_last_trade > 60:
            new_signal = -abs(current_size) * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~20 days on 4h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and fisher[i] < -1.0:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and fisher[i] > 1.0:
                new_signal = -abs(current_size) * 0.4
            elif is_range_market and fisher[i] < -1.5:
                new_signal = current_size * 0.35
            elif is_range_market and fisher[i] > 1.5:
                new_signal = -abs(current_size) * 0.35
        
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
        # Exit long when Fisher goes extreme overbought
        fisher_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher[i] > 2.0:
                fisher_reversal = True
            if position_side < 0 and fisher[i] < -2.0:
                fisher_reversal = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if regime shifts to strong trend against position
            if position_side > 0 and is_trend_market and trend_1d_bearish:
                regime_reversal = True
            if position_side < 0 and is_trend_market and trend_1d_bullish:
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