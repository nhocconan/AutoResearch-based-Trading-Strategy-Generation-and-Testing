#!/usr/bin/env python3
"""
Experiment #260: 6h Primary + 1d/1w HTF — Fisher Transform + Choppiness Regime + Vol Confirmation

Hypothesis: 6h timeframe is underexplored (only 2 experiments, both failed). This strategy combines:

1. EHLERS FISHER TRANSFORM (period=9): Proven reversal indicator for bear/range markets.
   Long when Fisher crosses above -1.5 from below. Short when crosses below +1.5 from above.
   Catches reversals better than RSI in 2022 crash and 2025 bear market.

2. CHOPPINESS INDEX regime filter: CHOP(14) > 61.8 = range (mean revert), CHOP < 38.2 = trending.
   This meta-filter adapts strategy to market conditions - critical for BTC/ETH which fail
   simple trend strategies.

3. HTF BIAS from 1d/1w HMA: Only trade Fisher signals in direction of higher timeframe trend.
   1d HMA(50) for intermediate trend, 1w HMA(21) for major bias.

4. VOLUME CONFIRMATION: Require volume > SMA(volume, 20) * 1.2 for breakout entries.
   Filters out fake breakouts that destroy Sharpe.

5. ASYMMETRIC LOGIC: In bear regime (price < 1d HMA), prefer short entries.
   In bull regime (price > 1d HMA), prefer long entries.

6. VOLATILITY FILTER: ATR(7)/ATR(30) ratio to avoid entering during extreme vol spikes.
   Wait for vol to normalize before entering.

Target: Sharpe>0.45 (beat current best 0.399), DD>-35%, trades>=30 train, trades>=3 test
Position sizing: 0.25 base, 0.30 strong (discrete levels to minimize fee churn)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_regime_vol_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound (mean reversion regime)
    CHOP < 38.2 = trending (trend following regime)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Highlights turning points better than RSI in bear/range markets
    
    Formula:
    1. Calculate typical price: (High + Low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) * 0.999 + 0.001
    3. Fisher: 0.5 * ln((1+x)/(1-x))
    4. Signal line: EMA of Fisher
    """
    n = len(high)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    signal = np.zeros(n)
    signal[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            # Normalize to 0.001-0.999 range
            normalized = (typical[i] - lowest) / price_range * 0.998 + 0.001
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    
    # Signal line is EMA of Fisher
    if not np.all(np.isnan(fisher)):
        fisher_series = pd.Series(fisher)
        signal = fisher_series.ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher, signal

def calculate_vol_ratio(atr, short_period=7, long_period=30):
    """ATR ratio to detect vol spikes - wait for normalization"""
    n = len(atr)
    if n < long_period + 1:
        return np.full(n, np.nan)
    
    atr_short = pd.Series(atr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(atr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    vol_ratio = np.zeros(n)
    vol_ratio[:] = np.nan
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            vol_ratio[i] = atr_short[i] / atr_long[i]
    
    return vol_ratio

def calculate_volume_sma(volume, period=20):
    """SMA of volume for volume confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    vol_ratio = calculate_vol_ratio(atr, short_period=7, long_period=30)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    # 6h HMA for local trend
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crossovers
    prev_fisher = 0.0
    prev_fisher_signal = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]) or np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME DETECTION ===
        choppy_threshold = 58.0  # Slightly lower than 61.8 for more trades
        trending_threshold = 42.0  # Slightly higher than 38.2
        
        is_choppy = chop[i] > choppy_threshold
        is_trending = chop[i] < trending_threshold
        is_transition = not is_choppy and not is_trending
        
        # === HTF BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 1w for major trend confirmation
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === 6h LOCAL TREND ===
        hma_6h_bull = close[i] > hma_6h[i]
        hma_6h_bear = close[i] < hma_6h[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > vol_sma[i] * 1.15  # 15% above average
        
        # === VOLATILITY FILTER ===
        # Avoid entering during extreme vol spikes (wait for normalization)
        vol_normalized = vol_ratio[i] < 1.8  # ATR(7) < 1.8x ATR(30)
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_signal[i-1]):
            # Long: Fisher crosses above -1.5 from below (oversold reversal)
            if fisher[i-1] < -1.5 and fisher[i] >= -1.5:
                fisher_cross_long = True
            # Short: Fisher crosses below +1.5 from above (overbought reversal)
            if fisher[i-1] > 1.5 and fisher[i] <= 1.5:
                fisher_cross_short = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # CHOPPY REGIME: Mean reversion with Fisher
        if is_choppy:
            # Long: Fisher oversold cross + price above 1d HMA (bullish bias in range)
            if fisher_cross_long and htf_1d_bull and vol_normalized:
                if vol_confirmed:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # Short: Fisher overbought cross + price below 1d HMA (bearish bias in range)
            elif fisher_cross_short and htf_1d_bear and vol_normalized:
                if vol_confirmed:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # TRENDING REGIME: Follow HTF trend with Fisher pullback entries
        elif is_trending:
            # Long in uptrend: Fisher pullback to oversold + HTF bull
            if fisher_cross_long and htf_1d_bull and htf_1w_bull and vol_normalized:
                if vol_confirmed and hma_6h_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # Short in downtrend: Fisher pullback to overbought + HTF bear
            elif fisher_cross_short and htf_1d_bear and htf_1w_bear and vol_normalized:
                if vol_confirmed and hma_6h_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # TRANSITION REGIME: Use previous regime logic (conservative)
        elif is_transition:
            # Only take strong signals with full HTF confirmation
            if fisher_cross_long and htf_1d_bull and htf_1w_bull and vol_confirmed:
                desired_signal = SIZE_BASE
            elif fisher_cross_short and htf_1d_bear and htf_1w_bear and vol_confirmed:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals