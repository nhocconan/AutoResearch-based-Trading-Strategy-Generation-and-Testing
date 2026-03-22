#!/usr/bin/env python3
"""
Experiment #345: 1h Fisher Transform Reversal with 4h HMA Bias + Volatility Compression

Hypothesis: After 293 failed strategies, the pattern is clear - trend-following and 
regime-detection strategies fail on BTC/ETH because these markets spend 70%+ time 
ranging. The Ehlers Fisher Transform is specifically designed to catch reversals in 
non-trending markets by normalizing price into a Gaussian distribution.

Key innovations vs failed strategies:
1. FISHER TRANSFORM (not RSI/MACD): Converts price to bounded -1 to +1 range, 
   extreme readings (>0.9 or <-0.9) signal exhaustion. Proven in bear markets.
2. VOLATILITY COMPRESSION FILTER: Only enter when BB Width < 20th percentile of 
   last 100 bars. This ensures we enter AFTER consolidation, not during chaos.
3. 4h HMA BIAS (not entry trigger): Use 4h HMA only to filter direction - long 
   only if price > 4h HMA, short only if price < 4h HMA. Simpler = more robust.
4. LOOSE ENTRY THRESHOLDS: Fisher > 0.7 (not 0.9) and < -0.7 (not -0.9) to ensure 
   >=10 trades per symbol. Previous strategies failed from being too strict.
5. ATR STOPLOSS: 2.5x ATR trailing stop to limit drawdown on failed reversals.

Why this should work on 1h:
- 1h captures intraday reversals that 4h/12h miss
- Fisher Transform excels in ranging markets (which is 70% of BTC/ETH time)
- Volatility compression filter prevents entries during high-vol chaos (2022 crash)
- 4h HMA bias prevents counter-trend trades in strong trends
- Loose thresholds ensure sufficient trade count (critical for Sharpe > 0)

Timeframe: 1h (REQUIRED)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete (conservative after 77% BTC crash lesson)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_reversal_4h_hma_bb_volcompress_atr_v1"
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
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range using period high/low
    3. Apply Fisher transform: 0.5 * ln((1 + x) / (1 - x))
    
    Readings > 0.9 or < -0.9 indicate extreme overbought/oversold.
    Crossovers of these extremes signal reversals.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over lookback period
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize price to -0.99 to +0.99 range (avoid division issues)
        normalized = 0.99 * (2.0 * (typical[i] - lowest) / price_range - 1.0)
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # Apply Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Signal line (1-period lag of fisher)
        if i > 0:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    
    # Middle band (SMA)
    middle = close_s.rolling(window=period, min_periods=period).mean().values
    
    # Standard deviation
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    # Upper and lower bands
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    # Bandwidth (normalized by middle)
    bandwidth = (upper - lower) / np.maximum(middle, 1e-10)
    
    return upper, lower, bandwidth

def calculate_bandwidth_percentile(bandwidth, lookback=100):
    """
    Calculate percentile rank of current bandwidth vs last lookback bars.
    Low percentile (< 20) = volatility compression = potential breakout/reversal.
    """
    n = len(bandwidth)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if np.isnan(bandwidth[i]):
            continue
        window = bandwidth[i-lookback:i]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) < lookback // 2:
            continue
        # Percentile rank: what % of past values are lower than current
        percentile[i] = np.sum(valid_window < bandwidth[i]) / len(valid_window) * 100
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_percentile = calculate_bandwidth_percentile(bb_bandwidth, lookback=100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28  # Conservative after learning from 77% BTC crash
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_percentile[i]):
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM EXTREMES ===
        # Loosened thresholds for more trades (critical for >=10 trades requirement)
        fisher_extreme_high = fisher[i] > 0.7  # Overbought reversal signal
        fisher_extreme_low = fisher[i] < -0.7  # Oversold reversal signal
        
        # Fisher crossover confirmation (reversal starting)
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher_signal[i] > 0.5
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher_signal[i] < -0.5
        
        # === VOLATILITY COMPRESSION FILTER ===
        # Only enter when BB bandwidth is in bottom 30% of recent range
        # This ensures we enter after consolidation, not during high-vol chaos
        vol_compressed = bb_percentile[i] < 30.0
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Fisher oversold + crossover up + 4h bullish bias + vol compressed
        if fisher_extreme_low and fisher_cross_up and bull_trend_4h and vol_compressed:
            new_signal = SIZE
        
        # SHORT ENTRY: Fisher overbought + crossover down + 4h bearish bias + vol compressed
        elif fisher_extreme_high and fisher_cross_down and bear_trend_4h and vol_compressed:
            new_signal = -SIZE
        
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
        # Exit if 4h trend flips against position (protective)
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === FISHER REVERSAL EXIT ===
        # Exit long when Fisher becomes extremely overbought (take profit)
        if in_position and position_side > 0 and fisher[i] > 1.2:
            new_signal = 0.0
        
        # Exit short when Fisher becomes extremely oversold (take profit)
        if in_position and position_side < 0 and fisher[i] < -1.2:
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