#!/usr/bin/env python3
"""
Experiment #297: 1h Supertrend + 4h HMA Bias + Choppiness Regime Filter

Hypothesis: After 296 experiments, clear patterns emerge for 1h timeframe:
1. RSI pullbacks FAIL consistently (#285, #291 both negative Sharpe)
2. Mean reversion on lower TFs is catastrophic (#290 Sharpe=-31)
3. 4h Supertrend + 1d HMA works best (#292 Sharpe=0.485)
4. Complex ensembles underperform simple trend following
5. ADX helps but shouldn't be too restrictive

This strategy combines DIFFERENT elements than failed 1h attempts:
1. 4h HMA(21) for directional bias (proven edge, stronger than 1d for 1h entries)
2. 1h Supertrend(10, 3.0) for clean trend signals (better than EMA crossover)
3. Choppiness Index(14) regime filter: CHOP>61.8=range(no trade), CHOP<38.2=trend(trade)
4. Fisher Transform(9) for entry timing (worked in #293, better than RSI)
5. ATR(14) trailing stoploss at 3.0*ATR (tighter than 12h strategies)
6. Volume confirmation: taker_buy_volume ratio > 0.55 for longs

Why this might beat #292 (4h Supertrend):
- 1h captures more trend moves than 4h (earlier entries)
- Choppiness filter avoids whipsaw in range markets (2022 bottom, 2025 bear)
- 4h HMA bias is more responsive than 1d for 1h entries
- Fisher Transform catches reversals earlier than EMA alone
- Volume confirmation reduces false breakouts

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 3.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_supertrend_4h_hma_chop_fisher_volume_v1"
timeframe = "1h"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_line, supertrend_direction (1=bullish, -1=bearish)
    
    Formula:
    1. ATR(period)
    2. Upper Band = (high + low)/2 + multiplier * ATR
    3. Lower Band = (high + low)/2 - multiplier * ATR
    4. Supertrend = Lower Band if bullish, Upper Band if bearish
    5. Direction flips when price crosses supertrend line
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    direction[:] = np.nan
    
    # Initialize
    supertrend[0] = lower_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if direction[i-1] == 1:
            # Previously bullish
            if close[i] < supertrend[i-1]:
                # Flip to bearish
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                # Stay bullish, use max of lower bands
                direction[i] = 1
                supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            # Previously bearish
            if close[i] > supertrend[i-1]:
                # Flip to bullish
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                # Stay bearish, use min of upper bands
                direction[i] = -1
                supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    return supertrend, direction

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market (avoid trend trades)
    CHOP < 38.2 = trending market (take trend trades)
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    n = len(close)
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high > lowest_low and atr_sum > 0:
            choppiness[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - transforms price into Gaussian distribution
    for clearer reversal signals. Period=9 is standard.
    """
    n = len(high)
    typical = (high + low) / 2.0
    
    normalized = np.zeros(n)
    normalized[:] = np.nan
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        if highest > lowest:
            normalized[i] = 0.66 * ((typical[i] - lowest) / (highest - lowest) - 0.5)
        else:
            normalized[i] = 0.0
    
    normalized = np.clip(normalized, -0.99, 0.99)
    
    norm_s = pd.Series(normalized)
    smoothed = norm_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    for i in range(period, n):
        if np.abs(smoothed[i]) < 0.99:
            fisher[i] = 0.5 * np.log((1 + smoothed[i]) / (1 - smoothed[i]))
    
    signal = np.roll(fisher, 1)
    signal[0] = np.nan
    
    return fisher, signal

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend_line, supertrend_dir = calculate_supertrend(high, low, close, 10, 3.0)
    choppiness = calculate_choppiness_index(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    
    # Volume ratio: taker buy / total volume
    volume_ratio = np.zeros(n)
    volume_ratio[:] = np.nan
    for i in range(n):
        if volume[i] > 0:
            volume_ratio[i] = taker_buy_volume[i] / volume[i]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.28  # Base position size (conservative)
    SIZE_REDUCED = 0.18  # Reduced size in high vol/choppy
    SIZE_INCREASED = 0.35  # Increased size in strong trend
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend_dir[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(choppiness[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(volume_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = directional bias (proven edge from #292)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (Choppiness Index) ===
        # CHOP > 61.8 = range (avoid trend trades, reduce size)
        # CHOP < 38.2 = trending (take trend trades, full size)
        # 38.2 <= CHOP <= 61.8 = transition (reduced size)
        choppy_market = choppiness[i] > 61.8
        trending_market = choppiness[i] < 38.2
        
        # === SUPERTREND SIGNAL ===
        # Supertrend direction: 1 = bullish, -1 = bearish
        supertrend_bullish = supertrend_dir[i] == 1
        supertrend_bearish = supertrend_dir[i] == -1
        
        # === FISHER TRANSFORM CONFIRMATION ===
        # Fisher crossing above signal line = bullish momentum
        fisher_bullish = fisher[i] > fisher_signal[i] and fisher_signal[i] < 0.0
        # Fisher crossing below signal line = bearish momentum
        fisher_bearish = fisher[i] < fisher_signal[i] and fisher_signal[i] > 0.0
        
        # === VOLUME CONFIRMATION ===
        # Taker buy ratio > 0.55 = buying pressure (for longs)
        # Taker buy ratio < 0.45 = selling pressure (for shorts)
        volume_bullish = volume_ratio[i] > 0.55
        volume_bearish = volume_ratio[i] < 0.45
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when ATR is elevated (>1.5x recent average)
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on regime and volatility
        if choppy_market or high_volatility:
            position_size = SIZE_REDUCED
        elif trending_market and not high_volatility:
            position_size = SIZE_INCREASED
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 4h bias up + Supertrend bullish + Fisher/volume confirmation
        # Looser in trending market, stricter in choppy
        if bull_trend_4h and supertrend_bullish:
            if trending_market:
                # Trending market: need Fisher OR volume confirmation
                if fisher_bullish or volume_bullish:
                    new_signal = position_size
            else:
                # Choppy/transition: need BOTH Fisher AND volume
                if fisher_bullish and volume_bullish:
                    new_signal = position_size
        
        # SHORT ENTRY: Mirror of long
        if bear_trend_4h and supertrend_bearish:
            if trending_market:
                # Trending market: need Fisher OR volume confirmation
                if fisher_bearish or volume_bearish:
                    new_signal = -position_size
            else:
                # Choppy/transition: need BOTH Fisher AND volume
                if fisher_bearish and volume_bearish:
                    new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 3.0 * ATR below highest close
                stoploss_price = highest_close - 3.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 3.0 * ATR above lowest close
                stoploss_price = lowest_close + 3.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
        # === SUPERTREND REVERSAL EXIT ===
        # Exit if Supertrend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and supertrend_bearish:
                new_signal = 0.0  # Supertrend flipped against long
            if position_side < 0 and supertrend_bullish:
                new_signal = 0.0  # Supertrend flipped against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals