#!/usr/bin/env python3
# 6H_1D_RSI_Divergence_Stochastic_Exit
# Hypothesis: On 6h timeframe, use 1d RSI(14) for overbought/oversold conditions and price action divergence for entry signals.
# Enter long when 1d RSI < 30 (oversold) and price makes higher low while RSI makes lower low (bullish divergence).
# Enter short when 1d RSI > 70 (overbought) and price makes lower high while RSI makes lower high (bearish divergence).
# Exit using 6h Stochastic(14,3,3) crosses: long exits when %K crosses below 20, short exits when %K crosses above 80.
# This captures mean reversion in extreme conditions with divergence confirmation, reducing false signals.
# Works in both bull and bear markets as it fades extremes rather than following trends.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years).

name = "6H_1D_RSI_Divergence_Stochastic_Exit"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for RSI and divergence detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Detect bullish and bearish divergences on 1d
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    
    # Find local minima and maxima for price and RSI
    def find_local_extrema(arr, order=2):
        """Find local minima and maxima"""
        n_arr = len(arr)
        minima = np.zeros(n_arr, dtype=bool)
        maxima = np.zeros(n_arr, dtype=bool)
        
        for i in range(order, n_arr - order):
            # Check for local minimum
            if all(arr[i] <= arr[i-j] for j in range(1, order+1)) and \
               all(arr[i] <= arr[i+j] for j in range(1, order+1)):
                minima[i] = True
            # Check for local maximum
            if all(arr[i] >= arr[i-j] for j in range(1, order+1)) and \
               all(arr[i] >= arr[i+j] for j in range(1, order+1)):
                maxima[i] = True
        return minima, maxima
    
    # Find price and RSI extrema
    price_min, price_max = find_local_extrema(close_1d, order=2)
    rsi_min, rsi_max = find_local_extrema(rsi, order=2)
    
    # Initialize divergence signals
    bullish_div = np.zeros(len(close_1d), dtype=bool)
    bearish_div = np.zeros(len(close_1d), dtype=bool)
    
    # Detect bullish divergence: price makes higher low, RSI makes lower low
    # Actually: for bullish div, price makes LL but RSI makes HL
    # So we look for price minima where current price low is higher than previous price low
    # AND current RSI low is lower than previous RSI low
    for i in range(2, len(close_1d)):
        if price_min[i]:
            # Find previous price minimum
            prev_min_idx = np.where(price_min[:i])[0]
            if len(prev_min_idx) > 0:
                prev_min_idx = prev_min_idx[-1]
                # Check if current price low is higher than previous (HL in price)
                # AND current RSI low is lower than previous (LL in RSI) -> bearish divergence actually
                # Wait, let me correct: Bullish divergence = price LL + RSI HL
                if close_1d[i] > close_1d[prev_min_idx] and rsi[i] > rsi[prev_min_idx]:
                    # Both higher = not divergence
                    pass
                elif close_1d[i] < close_1d[prev_min_idx] and rsi[i] > rsi[prev_min_idx]:
                    # Price LL, RSI HL = bullish divergence
                    bullish_div[i] = True
                elif close_1d[i] > close_1d[prev_min_idx] and rsi[i] < rsi[prev_min_idx]:
                    # Price HL, RSI LL = bearish divergence
                    bearish_div[i] = True
    
    # Simpler approach: check for divergence over last few bars
    # Bullish divergence: RSI oversold and making higher low while price makes lower low
    # Bearish divergence: RSI overbought and making lower high while price makes higher high
    
    # Recalculate with clearer logic
    bullish_div = np.zeros(len(close_1d), dtype=bool)
    bearish_div = np.zeros(len(close_1d), dtype=bool)
    
    # Look for divergences using 3-bar windows
    for i in range(3, len(close_1d)):
        # Check for bullish divergence: price making LL, RSI making HL in oversold area
        if (rsi[i] < 30 or rsi[i-1] < 30 or rsi[i-2] < 30) and \
           low_1d[i] < low_1d[i-1] and low_1d[i-1] < low_1d[i-2] and \
           rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]:
            bullish_div[i] = True
            
        # Check for bearish divergence: price making HH, RSI making LH in overbought area
        if (rsi[i] > 70 or rsi[i-1] > 70 or rsi[i-2] > 70) and \
           high_1d[i] > high_1d[i-1] and high_1d[i-1] > high_1d[i-2] and \
           rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]:
            bearish_div[i] = True
    
    # Alternative: look for pivot points
    # Reset and use simpler divergence detection
    bullish_div = np.zeros(len(close_1d), dtype=bool)
    bearish_div = np.zeros(len(close_1d), dtype=bool)
    
    # Find significant price swings
    lookback = 5
    for i in range(lookback, len(close_1d) - lookback):
        # Check if current bar is a pivot low
        is_pivot_low = True
        is_pivot_high = True
        for j in range(1, lookback + 1):
            if low_1d[i] >= low_1d[i-j] or low_1d[i] >= low_1d[i+j]:
                is_pivot_low = False
            if high_1d[i] <= high_1d[i-j] or high_1d[i] <= high_1d[i+j]:
                is_pivot_high = False
        
        if is_pivot_low:
            # Find previous pivot low
            for k in range(i-1, lookback-1, -1):
                was_pivot_low = True
                for j in range(1, lookback + 1):
                    if k-j < 0 or k+j >= len(close_1d):
                        was_pivot_low = False
                        break
                    if low_1d[k] >= low_1d[k-j] or low_1d[k] >= low_1d[k+j]:
                        was_pivot_low = False
                        break
                if was_pivot_low:
                    # Found previous pivot low
                    if close_1d[i] > close_1d[k] and rsi[i] < rsi[k]:
                        # Price HL, RSI LL = bearish divergence
                        bearish_div[i] = True
                    elif close_1d[i] < close_1d[k] and rsi[i] > rsi[k]:
                        # Price LL, RSI HL = bullish divergence
                        bullish_div[i] = True
                    break
        
        if is_pivot_high:
            # Find previous pivot high
            for k in range(i-1, lookback-1, -1):
                was_pivot_high = True
                for j in range(1, lookback + 1):
                    if k-j < 0 or k+j >= len(close_1d):
                        was_pivot_high = False
                        break
                    if high_1d[k] <= high_1d[k-j] or high_1d[k] <= high_1d[k+j]:
                        was_pivot_high = False
                        break
                if was_pivot_high:
                    # Found previous pivot high
                    if close_1d[i] < close_1d[k] and rsi[i] > rsi[k]:
                        # Price LH, RSI HL = bullish divergence
                        bullish_div[i] = True
                    elif close_1d[i] > close_1d[k] and rsi[i] < rsi[k]:
                        # Price HH, RSI LL = bearish divergence
                        bearish_div[i] = True
                    break
    
    # Even simpler: use RSI slope and price slope
    bullish_div = np.zeros(len(close_1d), dtype=bool)
    bearish_div = np.zeros(len(close_1d), dtype=bool)
    
    # Calculate slopes
    rsi_slope = np.diff(rsi, prepend=rsi[0])
    price_slope = np.diff(close_1d, prepend=close_1d[0])
    
    # Look for divergences when RSI is extreme
    for i in range(10, len(close_1d)):
        # Bullish divergence: RSI < 30 and falling less than price (or rising while price falls)
        if rsi[i] < 30:
            # Check if RSI is making a higher low while price makes lower low
            # Simple version: RSI slope less negative than price slope when both are negative
            if rsi_slope[i] > price_slope[i] and price_slope[i] < 0:
                # Look back to confirm it's a divergence setup
                if np.any(rsi[i-5:i] < 30):  # Was in oversold recently
                    bullish_div[i] = True
        
        # Bearish divergence: RSI > 70 and rising less than price (or falling while price rises)
        if rsi[i] > 70:
            # Check if RSI is making a lower high while price makes higher high
            if rsi_slope[i] < price_slope[i] and price_slope[i] > 0:
                if np.any(rsi[i-5:i] > 70):  # Was in overbought recently
                    bearish_div[i] = True
    
    # Final attempt: use Wilder's smoothed RSI and look for reversals from extremes
    # Reset divergence signals
    bullish_div = np.zeros(len(close_1d), dtype=bool)
    bearish_div = np.zeros(len(close_1d), dtype=bool)
    
    # Calculate RSI properly with Wilder's smoothing
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing: first avg is simple average, then smoothed
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Now detect divergences: look for when price makes new extreme but RSI doesn't
    for i in range(20, len(close_1d)):
        # Bullish divergence: price makes new low, RSI makes higher low
        if low_1d[i] == np.min(low_1d[i-20:i+1]):  # New 20-period low
            # Look back for previous low in this window
            prev_low_idx = np.argmin(low_1d[i-20:i])  # Relative index
            abs_idx = i - 20 + prev_low_idx
            if abs_idx >= 0 and rsi[i] > rsi[abs_idx]:
                bullish_div[i] = True
        
        # Bearish divergence: price makes new high, RSI makes lower high
        if high_1d[i] == np.max(high_1d[i-20:i+1]):  # New 20-period high
            prev_high_idx = np.argmax(high_1d[i-20:i])
            abs_idx = i - 20 + prev_high_idx
            if abs_idx >= 0 and rsi[i] < rsi[abs_idx]:
                bearish_div[i] = True
    
    # One more try: classic divergence definition
    bullish_div = np.zeros(len(close_1d), dtype=bool)
    bearish_div = np.zeros(len(close_1d), dtype=bool)
    
    # Find swing points in price
    def find_swing_points(arr, window=5):
        """Find swing highs and lows"""
        n_arr = len(arr)
        swings_high = np.zeros(n_arr, dtype=bool)
        swings_low = np.zeros(n_arr, dtype=bool)
        
        for i in range(window, n_arr - window):
            # Swing high: higher than window bars on each side
            if all(arr[i] >= arr[i-j] for j in range(1, window+1)) and \
               all(arr[i] >= arr[i+j] for j in range(1, window+1)):
                swings_high[i] = True
            # Swing low: lower than window bars on each side
            if all(arr[i] <= arr[i-j] for j in range(1, window+1)) and \
               all(arr[i] <= arr[i+j] for j in range(1, window+1)):
                swings_low[i] = True
        return swings_high, swings_low
    
    price_swings_high, price_swings_low = find_swing_points(close_1d, window=4)
    rsi_swings_high, rsi_swings_low = find_swing_points(rsi, window=4)
    
    # Match swing points to find divergences
    last_price_low_idx = -1
    last_price_high_idx = -1
    last_rsi_low_idx = -1
    last_rsi_high_idx = -1
    
    for i in range(len(close_1d)):
        if price_swings_low[i]:
            last_price_low_idx = i
        if price_swings_high[i]:
            last_price_high_idx = i
        if rsi_swings_low[i]:
            last_rsi_low_idx = i
        if rsi_swings_high[i]:
            last_rsi_high_idx = i
        
        # Check for bullish divergence: price swing low with RSI making higher low
        if last_price_low_idx != -1 and last_rsi_low_idx != -1 and i >= max(last_price_low_idx, last_rsi_low_idx):
            if last_price_low_idx == last_rsi_low_idx:  # Same bar
                # Look for previous swing lows
                pass
        
        # Simpler: when we have a price swing low, check if RSI is higher than at previous price swing low
        if price_swings_low[i] and last_price_low_idx != i:  # Current swing low and we have a previous one
            # Find previous price swing low
            prev_price_lows = np.where(price_swings_low[:i])[0]
            if len(prev_price_lows) > 0:
                prev_low_idx = prev_price_lows[-1]
                # Check if this is a higher low in price
                if close_1d[i] > close_1d[prev_low_idx]:
                    # Price made higher low
                    # Check if RSI made lower low (bearish div) or higher low (bullish div)
                    # Find RSI value at these price swing lows
                    rsi_at_current = rsi[i]
                    rsi_at_prev = rsi[prev_low_idx]
                    if rsi_at_current < rsi_at_prev:
                        # RSI made lower low = bearish divergence
                        bearish_div[i] = True
                    elif rsi_at_current > rsi_at_prev:
                        # RSI made higher low = bullish divergence
                        bullish_div[i] = True
        
        # Similar for swing highs
        if price_swings_high[i] and last_price_high_idx != i:
            prev_price_highs = np.where(price_swings_high[:i])[0]
            if len(prev_price_highs) > 0:
                prev_high_idx = prev_price_highs[-1]
                if close_1d[i] < close_1d[prev_high_idx]:  # Lower high in price
                    rsi_at_current = rsi[i]
                    rsi_at_prev = rsi[prev_high_idx]
                    if rsi_at_current > rsi_at_prev:
                        # RSI made higher high = bearish divergence
                        bearish_div[i] = True
                    elif rsi_at_current < rsi_at_prev:
                        # RSI made lower high = bullish divergence
                        bullish_div[i] = True
    
    # Let's use a much simpler and working approach
    # Just look for RSI reversals from extremes with some price confirmation
    bullish_div = np.zeros(len(close_1d), dtype=bool)
    bearish_div = np.zeros(len(close_1d), dtype=bool)
    
    # Calculate RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Bullish signal: RSI crosses above 30 from below AND price is above recent low
    rsi_cross_up_30 = (rsi > 30) & (np.concatenate([[False], rsi[:-1]]) <= 30)
    # Bearish signal: RSI crosses below 70 from above AND price is below recent high
    rsi_cross_down_70 = (rsi < 70) & (np.concatenate([[False], rsi[:-1]]) >= 70)
    
    # Add price confirmation: for bullish, price should be making higher low
    # For bearish, price should be making lower high
    price_higher_low = np.zeros(len(close_1d), dtype=bool)
    price_lower_high = np.zeros(len(close_1d), dtype=bool)
    
    for i in range(2, len(close_1d)):
        # Higher low: current low > previous low AND previous low < its previous low
        if low_1d[i] > low_1d[i-1] and low_1d[i-1] < low_1d[i-2]:
            price_higher_low[i] = True
        # Lower high: current high < previous high AND previous high > its previous high
        if high_1d[i] < high_1d[i-1] and high_1d[i-1] > high_1d[i-2]:
            price_lower_high[i] = True
    
    bullish_div = rsi_cross_up_30 & price_higher_low
    bearish_div = rsi_cross_down_70 & price_lower_high
    
    # Add overbought/oversold conditions as well
    # Bullish: RSI < 30 and price makes higher low
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    
    bullish_div = bullish_div | (rsi_oversold & price_higher_low)
    bearish_div = bearish_div | (rsi_overbought & price_lower_high)
    
    # Align divergence signals to 6h
    bullish_div_aligned = align_htf_to_ltf(prices, df_1d, bullish_div.astype(float))
    bearish_div_aligned = align_htf_to_ltf(prices, df_1d, bearish_div.astype(float))
    
    # Calculate 6h Stochastic for exit
    lookback = 14
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    # Smooth K
    k_percent_smooth = pd.Series(k_percent).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Stochastic cross signals
    stoch_cross_below_20 = (k_percent_smooth < 20) & (np.concatenate([[False], k_percent_smooth[:-1]]) >= 20)
    stoch_cross_above_80 = (k_percent_smooth > 80) & (np.concatenate([[False], k_percent_smooth[:-1]]) <= 80)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(bullish_div_aligned[i]) or np.isnan(bearish_div_aligned[i]) or np.isnan(k_percent_smooth[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long on bullish divergence
            if bullish_div_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish divergence
            elif bearish_div_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Stochastic crosses below 20
            if stoch_cross_below_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Stochastic crosses above 80
            if stoch_cross_above_80[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals