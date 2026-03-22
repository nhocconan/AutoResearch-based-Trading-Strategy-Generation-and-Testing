#!/usr/bin/env python3
"""
Experiment #419: 12h Fisher Transform + Choppiness Regime + 1d HMA Trend

Hypothesis: After 418 failed experiments, the key insight is that 12h timeframe
needs REVERSAL-BASED entries (not breakout-based). 12h bars capture significant
price moves, making them ideal for catching trend reversals rather than breakouts.

This strategy combines:

1. EHLERS FISHER TRANSFORM (period=9) - Primary entry signal
   - Fisher crosses above -1.5 from below → Long signal
   - Fisher crosses below +1.5 from above → Short signal
   - Fisher excels at catching reversals in bear/range markets (2022, 2025)
   - More responsive than RSI for 12h timeframe

2. CHOPPINESS INDEX (period=14) - Regime filter
   - CHOP > 61.8 = ranging market (use Fisher reversals)
   - CHOP < 38.2 = trending market (use Fisher with trend bias only)
   - 38.2-61.8 = neutral (reduce position size or stay flat)
   - This is the META-FILTER that determines which Fisher signals to take

3. 1d HMA(21) TREND BIAS (via mtf_data helper)
   - In trending regime: only take Fisher signals WITH trend direction
   - In ranging regime: take Fisher signals in EITHER direction (mean reversion)
   - HMA smoother than EMA, critical for 12h/1d alignment

4. ATR(14) TRAILING STOP at 2.5x
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from 2022-style crashes

5. POSITION SIZING: 0.25 discrete (conservative for 12h)
   - Max 25% capital per position
   - Discrete levels minimize fee churn on slower timeframe
   - In neutral regime: reduce to 0.15

Why this should work on 12h:
- Fisher Transform catches reversals (better than breakout on 12h)
- Choppiness filter avoids whipsaws in unclear markets
- 1d HMA provides trend bias without lag
- Fewer trades (~30-50/year) = less fee drag
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels (0.15 in neutral regime)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_chop_regime_1d_hma_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
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
    Transforms price into a Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0 if 'close' in dir() else (high + low) / 2.0
    
    # Use close for Fisher if available
    close_arr = high  # placeholder, will be replaced
    typical = (high + low + close_arr) / 3.0
    
    # Normalize price to -1 to +1 range
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest == lowest:
            continue
        
        # Normalize to -1 to +1
        normalized = 2.0 * ((close_arr[i] - lowest) / (highest - lowest)) - 1.0
        normalized = max(-0.999, min(0.999, normalized))  # Clamp to avoid log errors
        
        # Apply Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        if i > period:
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform (simplified version).
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    
    for i in range(period, n):
        # Find highest high and lowest low over period using close as proxy
        highest = close[i-period+1:i+1].max()
        lowest = close[i-period+1:i+1].min()
        
        if highest == lowest:
            fisher[i] = fisher[i-1] if i > period else 0.0
            continue
        
        # Normalize to -1 to +1
        normalized = 2.0 * ((close[i] - lowest) / (highest - lowest)) - 1.0
        normalized = max(-0.999, min(0.999, normalized))
        
        # Apply Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    return fisher

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest == lowest:
            chop[i] = 50.0
            continue
        
        # Sum of ATR over period
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr_sum += max(tr1, tr2, tr3)
        
        if tr_sum == 0:
            chop[i] = 50.0
            continue
        
        # CHOP formula
        chop[i] = 100 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher(close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_TREND = 0.25  # Full size in clear regimes
    SIZE_NEUTRAL = 0.15  # Reduced size in neutral regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        neutral_market = not ranging_market and not trending_market
        
        # === 1d HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Need previous fisher value for crossover detection
        fisher_prev = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else fisher[i]
        
        # Fisher crossover signals
        fisher_long_signal = (fisher_prev < -1.5 and fisher[i] >= -1.5)  # Cross above -1.5
        fisher_short_signal = (fisher_prev > 1.5 and fisher[i] <= 1.5)   # Cross below +1.5
        
        # Also check for extreme reversals
        fisher_extreme_long = fisher[i] < -2.0  # Very oversold
        fisher_extreme_short = fisher[i] > 2.0  # Very overbought
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        current_size = SIZE_TREND if not neutral_market else SIZE_NEUTRAL
        
        # RANGING REGIME: Take Fisher reversals in EITHER direction (mean reversion)
        if ranging_market:
            if fisher_long_signal or fisher_extreme_long:
                new_signal = current_size
            elif fisher_short_signal or fisher_extreme_short:
                new_signal = -current_size
        
        # TRENDING REGIME: Only take Fisher signals WITH trend direction
        elif trending_market:
            if bull_trend_1d and (fisher_long_signal or fisher_extreme_long):
                new_signal = current_size
            elif bear_trend_1d and (fisher_short_signal or fisher_extreme_short):
                new_signal = -current_size
        
        # NEUTRAL REGIME: Only take extreme Fisher signals (reduce size)
        elif neutral_market:
            if fisher_extreme_long:
                new_signal = SIZE_NEUTRAL
            elif fisher_extreme_short:
                new_signal = -SIZE_NEUTRAL
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 1d trend turns bearish, exit short if 1d trend turns bullish
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d and trending_market:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d and trending_market:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals