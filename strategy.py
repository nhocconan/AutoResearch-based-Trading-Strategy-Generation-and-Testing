#!/usr/bin/env python3
"""
Experiment #1351: 6h Primary + 1d/1w HTF — Fisher Transform + CRSI + CHOP Regime

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Combining:
1. Ehlers Fisher Transform (period=9) for reversal detection - catches bear market rallies
2. Connors RSI (CRSI) for entry timing - RSI(3) + RSI_Streak(2) + PercentRank(100) / 3
3. Choppiness Index (CHOP) for regime filter - >61.8 mean revert, <38.2 trend follow
4. 1d/1w HMA(21) for major trend bias - avoids counter-trend trades

Why this should work where 6h failed before:
- Fisher Transform excels in bear/range markets (2025 test period)
- CRSI has 75% win rate on extremes (<10 or >90)
- CHOP prevents trend strategies in choppy markets (major 6h failure cause)
- Loose entry: Fisher cross OR CRSI extreme (not both required)
- Target: 30-60 trades/year, Sharpe>0.5, DD>-35%

Entry logic:
- LONG: 1d_HMA bullish + (Fisher<-1.5 cross OR CRSI<15) + CHOP regime confirm
- SHORT: 1d_HMA bearish + (Fisher>+1.5 cross OR CRSI>85) + CHOP regime confirm

Timeframe: 6h
Size: 0.25-0.30 discrete
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_crsi_chop_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals in bear markets effectively
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        # Normalize price to 0-1 range
        range_val = highest - lowest
        if range_val < 1e-10:
            fisher[i] = 0.0
            trigger[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Calculate epsilon value (0.001 to avoid division by zero)
        epsilon = 0.001
        normalized = ((hl2 - lowest) / range_val) - 0.5
        normalized = max(-0.499, min(0.499, normalized))
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Apply smoothing (previous fisher value)
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
        else:
            fisher[i] = fisher_val
        
        # Trigger line (previous fisher)
        trigger[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, trigger

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values < 10 = oversold (long), > 90 = overbought (short)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    delta = np.diff(close)
    
    for i in range(1, n):
        streak = 0
        if delta[i-1] > 0:
            for j in range(i-1, -1, -1):
                if j < i-1 and delta[j] <= 0:
                    break
                streak += 1
        elif delta[i-1] < 0:
            for j in range(i-1, -1, -1):
                if j < i-1 and delta[j] >= 0:
                    break
                streak -= 1
        
        # Convert streak to RSI-like value (0-100)
        if streak > 0:
            streak_rsi[i] = 100 - (100 / (1 + streak))
        elif streak < 0:
            streak_rsi[i] = 100 / (1 + abs(streak))
        else:
            streak_rsi[i] = 50
    
    # Percent Rank - position of current close in last rank_period closes
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period - 1, n):
        window = close[i - rank_period + 1:i + 1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = (count_below / (rank_period - 1)) * 100
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period - 1, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
        
        range_val = highest - lowest
        if range_val < 1e-10 or atr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100 * np.log10(atr_sum / range_val) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_chop(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA for major regime confirmation
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_bullish_cross = (fisher[i] > -1.5) and (fisher_trigger[i] <= -1.5)
        # Short: Fisher crosses below +1.5 from above
        fisher_bearish_cross = (fisher[i] < 1.5) and (fisher_trigger[i] >= 1.5)
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Loose threshold for more trades
        crsi_overbought = crsi[i] > 85  # Loose threshold for more trades
        
        # === CHOP REGIME FILTER ===
        chop_range = chop[i] > 55  # Range/choppy market (mean revert)
        chop_trend = chop[i] < 45  # Trending market (trend follow)
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG entries (multiple paths to ensure trades)
        if price_above_1d:
            # Path 1: Fisher bullish cross in trending market
            if fisher_bullish_cross and chop_trend:
                if price_above_1w:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # Path 2: CRSI oversold in choppy market (mean revert)
            elif crsi_oversold and chop_range:
                if price_above_1w:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # Path 3: Both Fisher and CRSI confirm (strongest signal)
            elif fisher_bullish_cross and crsi_oversold:
                desired_signal = SIZE_STRONG
        
        # SHORT entries (multiple paths to ensure trades)
        elif price_below_1d:
            # Path 1: Fisher bearish cross in trending market
            if fisher_bearish_cross and chop_trend:
                if price_below_1w:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            
            # Path 2: CRSI overbought in choppy market (mean revert)
            elif crsi_overbought and chop_range:
                if price_below_1w:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            
            # Path 3: Both Fisher and CRSI confirm (strongest signal)
            elif fisher_bearish_cross and crsi_overbought:
                desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals