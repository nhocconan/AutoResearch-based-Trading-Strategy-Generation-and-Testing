#!/usr/bin/env python3
"""
Experiment #200: 1h Primary + 4h/12h HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: Lower timeframe (1h) needs looser entry conditions than 4h to generate trades.
Previous 1h strategies failed with 0 trades due to too many confluence requirements.

This experiment uses:
1. Ehlers Fisher Transform (period=9) - catches reversals with frequent signals
2. Choppiness Index (14) - regime detection (range >55, trend <45)
3. 4h HMA(21) - primary HTF trend direction
4. 12h HMA(50) - macro bias filter (optional, not required for entry)
5. Session filter (8-20 UTC) - liquidity hours only
6. Looser thresholds: Fisher crosses at -1.5/+1.5 (not extreme -2/+2)

Key changes from failed 1h strategies (#190, #195):
1. Fisher Transform instead of RSI/CRSI (more frequent crossover signals)
2. OR logic for some conditions (not ALL filters must agree)
3. Session filter is optional bonus, not required
4. Hold positions through minor pullbacks (don't exit on every signal change)
5. Asymmetric sizing: 0.25 standard, 0.35 with HTF confirmation

TARGET: 40-80 trades/year on 1h, Sharpe > 0.3 on ALL symbols
Position sizing: 0.0, ±0.25, ±0.35 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_hma_regime_4h12h_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    hma = (2 * close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean() 
           - close_s.ewm(span=period, min_periods=period, adjust=False).mean())
    hma = hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            tr_sum += tr
        
        range_val = highest - lowest
        if range_val < 1e-10 or tr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    
    Steps:
    1. Calculate typical price (HL2)
    2. Normalize to -1 to +1 range over lookback period
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    4. Smooth with EMA
    
    Long signal: Fisher crosses above -1.5 (from below)
    Short signal: Fisher crosses below +1.5 (from above)
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    # Typical price
    typical = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest and lowest over period
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            normalized = 0.0
        else:
            # Normalize to -1 to +1
            normalized = 2.0 * ((typical[i] - lowest) / range_val) - 1.0
            # Clamp to avoid division by zero in Fisher
            normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
        
        # Signal line (1-period EMA of fisher)
        if i > period:
            fisher_signal[i] = 0.7 * fisher[i] + 0.3 * fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate HTF HMA for trend direction (aligned properly)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_STANDARD = 0.25
    POSITION_SIZE_HIGH_CONV = 0.35
    
    # Track position state for stoploss and hold logic
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_fisher = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # Extract hour for session filter
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20  # UTC liquidity hours
        
        # === HTF TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # 4h HMA slope (trend strength)
        hma_4h_slope = hma_4h_aligned[i] - hma_4h_aligned[i-4] if i >= 4 else 0.0
        hma_4h_bullish = hma_4h_slope > 0.0
        hma_4h_bearish = hma_4h_slope < 0.0
        
        # === REGIME DETECTION (Choppiness) ===
        is_range = chop_14[i] > 55.0  # Ranging market
        is_trend = chop_14[i] < 45.0  # Trending market
        # Neutral zone 45-55: use trend logic
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 (bullish reversal)
        fisher_cross_up = (fisher_signal[i] > -1.5) and (fisher_signal[i-1] <= -1.5)
        # Fisher crosses below +1.5 (bearish reversal)
        fisher_cross_down = (fisher_signal[i] < 1.5) and (fisher_signal[i-1] >= 1.5)
        
        # Extreme Fisher levels (stronger signal)
        fisher_oversold = fisher_signal[i] < -1.8
        fisher_overbought = fisher_signal[i] > 1.8
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        entry_reason = ""
        
        if is_range:
            # MEAN REVERSION MODE - fade extremes
            # Long: Fisher oversold + RSI < 35
            if fisher_oversold or (fisher_cross_up and rsi_14[i] < 40):
                new_signal = POSITION_SIZE_STANDARD
                entry_reason = "range_long"
            
            # Short: Fisher overbought + RSI > 65
            elif fisher_overbought or (fisher_cross_down and rsi_14[i] > 60):
                new_signal = -POSITION_SIZE_STANDARD
                entry_reason = "range_short"
        
        elif is_trend:
            # TREND FOLLOWING MODE - pullback entries
            # Long: 4h HMA bullish + Fisher cross up from pullback
            if hma_4h_bullish and price_above_hma_4h:
                if fisher_cross_up or (fisher_signal[i] < -0.5 and fisher_signal[i-1] < fisher_signal[i]):
                    if hma_4h_aligned[i] > hma_12h_aligned[i]:  # 4h above 12h = strong trend
                        new_signal = POSITION_SIZE_HIGH_CONV
                        entry_reason = "trend_long_strong"
                    else:
                        new_signal = POSITION_SIZE_STANDARD
                        entry_reason = "trend_long"
            
            # Short: 4h HMA bearish + Fisher cross down from rally
            elif hma_4h_bearish and price_below_hma_4h:
                if fisher_cross_down or (fisher_signal[i] > 0.5 and fisher_signal[i-1] > fisher_signal[i]):
                    if hma_4h_aligned[i] < hma_12h_aligned[i]:  # 4h below 12h = strong trend
                        new_signal = -POSITION_SIZE_HIGH_CONV
                        entry_reason = "trend_short_strong"
                    else:
                        new_signal = -POSITION_SIZE_STANDARD
                        entry_reason = "trend_short"
        
        # === SESSION FILTER (optional bonus, not required) ===
        # Only reduce position if entering counter to session expectation
        # Don't block entries entirely (was causing 0 trades in #195)
        if new_signal != 0.0 and not in_session:
            # Outside session: reduce size by 20%
            new_signal = new_signal * 0.8
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and Fisher hasn't reversed yet
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if Fisher not overbought yet
                if fisher_signal[i] < 1.5:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if Fisher not oversold yet
                if fisher_signal[i] > -1.5:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 4h HMA (trend changed)
        if in_position and position_side > 0 and price_below_hma_4h and is_trend:
            new_signal = 0.0
        
        # Exit short if price crosses above 4h HMA (trend changed)
        if in_position and position_side < 0 and price_above_hma_4h and is_trend:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_fisher = fisher_signal[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_fisher = fisher_signal[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_fisher = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals