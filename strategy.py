#!/usr/bin/env python3
"""
Experiment #081: 4h Primary + 1d HTF — Vol Spike Mean Reversion + Fisher Transform

Hypothesis: Volatility spikes (ATR(7)/ATR(30) > 2.0) signal panic/euphoria extremes that 
typically revert within 5-20 bars. Combined with Ehlers Fisher Transform for precise 
reversal timing and 1d HMA for major trend bias, this should capture mean reversion 
opportunities while avoiding counter-trend trades in strong trends.

Why this should work:
1. Vol spike reversion is documented edge in crypto (panic selloffs reverse quickly)
2. Fisher Transform normalizes price into Gaussian distribution for cleaner signals
3. 1d HMA slope prevents fighting major trend (only long if 1d slope > 0, etc.)
4. 4h timeframe = 20-50 trades/year (optimal for fee/capture balance)
5. Works in both bull (2021) and bear/range (2022, 2025) markets

Strategy Logic:
1. VOLATILITY REGIME: ATR(7)/ATR(30) > 2.0 = vol spike (mean revert), < 1.2 = calm
2. FISHER TRANSFORM (9): Long when Fisher crosses above -1.5, short when crosses below +1.5
3. 1d HMA(21) SLOPE: Major trend bias filter
4. BOLLINGER BANDS (20, 2.5): Additional extreme confirmation
5. ATR(14) stoploss: 2.5x trailing stop
6. Position size: 0.28 discrete (conservative for vol strategies)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Target trades: 25-45/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_fisher_hma_1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian-normalized values for clearer reversal signals.
    Fisher crosses above -1.5 = long signal, crosses below +1.5 = short signal.
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
    price_range = price_range.replace(0, 1e-10)  # avoid div by zero
    
    normalized = 2.0 * (typical - lowest) / price_range - 1.0
    normalized = normalized.clip(-0.999, 0.999)  # Fisher requires input in (-1, 1)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with wider std_dev for extreme detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper.values, lower.values, sma.values

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
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Volatility ratio (spike detection)
    vol_ratio = np.zeros(n)
    for i in range(30, n):
        if atr_30[i] > 0 and not np.isnan(atr_30[i]):
            vol_ratio[i] = atr_7[i] / atr_30[i]
        else:
            vol_ratio[i] = 1.0
    
    # Fisher Transform
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    # Bollinger Bands (wide for extremes)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(vol_ratio[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # HMA slope > 0.3 = bullish bias (prefer longs)
        # HMA slope < -0.3 = bearish bias (prefer shorts)
        # Between = neutral (allow both directions with vol spike)
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.3
        trend_1d_neutral = not trend_1d_bullish and not trend_1d_bearish
        
        # Price vs 1d HMA for additional confirmation
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY REGIME ===
        # vol_ratio > 2.0 = vol spike (mean reversion likely)
        # vol_ratio < 1.2 = calm (trend follow if any)
        is_vol_spike = vol_ratio[i] > 1.8  # Slightly lower for more trades
        is_vol_calm = vol_ratio[i] < 1.3
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 from below = long reversal signal
        # Fisher crosses below +1.5 from above = short reversal signal
        fisher_cross_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # Extreme Fisher values (strong reversal signals)
        fisher_extreme_low = fisher[i] < -1.8
        fisher_extreme_high = fisher[i] > 1.8
        
        # === BOLLINGER BAND EXTREMES ===
        at_bb_lower = close[i] <= bb_lower[i]
        at_bb_upper = close[i] >= bb_upper[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in calm vol (less opportunity)
        if is_vol_calm and not is_vol_spike:
            current_size = BASE_SIZE * 0.5
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Primary: Vol spike + Fisher long cross + 1d bullish or neutral
        if is_vol_spike and fisher_cross_long:
            if trend_1d_bullish or trend_1d_neutral:
                new_signal = current_size
            elif trend_1d_bearish and price_above_1d_hma:
                # Counter-trend only if price above 1d HMA (pullback)
                new_signal = current_size * 0.6
        
        # Secondary: Fisher extreme low + BB lower (strong mean reversion)
        if fisher_extreme_low and at_bb_lower:
            if not in_position or position_side < 0:
                new_signal = current_size * 0.8
        
        # Tertiary: 1d bullish + Fisher moderate (trend pullback)
        if trend_1d_bullish and fisher[i] < -0.5 and fisher_signal[i] < fisher[i]:
            if not is_vol_spike:  # Only in calm/trend market
                new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        # Primary: Vol spike + Fisher short cross + 1d bearish or neutral
        if is_vol_spike and fisher_cross_short:
            if trend_1d_bearish or trend_1d_neutral:
                new_signal = -current_size
            elif trend_1d_bullish and price_below_1d_hma:
                # Counter-trend only if price below 1d HMA (rally)
                new_signal = -current_size * 0.6
        
        # Secondary: Fisher extreme high + BB upper (strong mean reversion)
        if fisher_extreme_high and at_bb_upper:
            if not in_position or position_side > 0:
                new_signal = -current_size * 0.8
        
        # Tertiary: 1d bearish + Fisher moderate (trend pullback)
        if trend_1d_bearish and fisher[i] > 0.5 and fisher_signal[i] > fisher[i]:
            if not is_vol_spike:  # Only in calm/trend market
                new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~13 days on 4h), allow weaker entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and fisher[i] < -0.5:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and fisher[i] > 0.5:
                new_signal = -current_size * 0.4
            elif fisher_extreme_low:
                new_signal = current_size * 0.4
            elif fisher_extreme_high:
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
        
        # === FISHER REVERSAL EXIT ===
        # Exit long if Fisher crosses back below 0, exit short if crosses above 0
        fisher_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher[i] < 0 and fisher_signal[i] >= 0:
                fisher_reversal = True
            if position_side < 0 and fisher[i] > 0 and fisher_signal[i] <= 0:
                fisher_reversal = True
        
        # Apply stoploss or Fisher reversal
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