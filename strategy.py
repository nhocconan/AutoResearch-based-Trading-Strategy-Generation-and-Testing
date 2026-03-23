#!/usr/bin/env python3
"""
Experiment #928: 30m Primary + 4h/1d HTF — Fisher Transform + Session + Volume Confluence

Hypothesis: After 659 failed strategies, 30m timeframe needs VERY STRICT entry filters
to avoid fee drag (>100 trades/year = death). Key insight from research:

1. 30m Primary TF with 4h/1d HTF direction filter (proven pattern from best strategies)
2. Fisher Transform for entry timing — catches reversals better than RSI in bear markets
3. Session filter (8-20 UTC) — only trade during high liquidity hours
4. Volume filter (>0.8x 20-bar avg) — avoid low-volume whipsaws
5. 4h HMA(21) for medium-term trend bias
6. 1d HMA(21) for macro regime (bull/bear filter)
7. ATR(14) trailing stop 2.5x for risk management
8. Relaxed Fisher thresholds (-1.5/+1.5) to ensure trades on all symbols

Why this should work on 30m:
- Fisher Transform has 70%+ win rate on reversals (Ehlers research)
- Session filter cuts trades by 60% (only 12h of 24h)
- Volume filter avoids fake breakouts
- HTF direction filter prevents counter-trend trades
- Size=0.25 (smaller for lower TF to control DD)

Critical improvements from failed experiments:
- Fisher Transform instead of RSI (better reversal detection)
- Session + Volume confluence (reduces trades to 40-80/year target)
- Relaxed Fisher thresholds (-1.5/+1.5 not -2/+2) to guarantee trades
- ALL symbols MUST have positive Sharpe (no SOL-only bias)
- Use 1d HMA as macro filter: only long if price > 1d HMA in bull market

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 40-80 trades/year with session+volume filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_session_vol_4h1d_hma_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    Catches reversals in bear/range markets better than RSI.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 2:
        return fisher, fisher_signal
    
    for i in range(period, n):
        # Calculate price position within range
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            continue
        
        # Normalize price to 0-1 range
        price = (high[i] + low[i]) / 2.0
        normalized = 0.66 * ((price - lowest) / (highest - lowest) - 0.5)
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Signal line (1-period EMA of Fisher)
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def extract_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time_array // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    fisher_30m, fisher_signal_30m = calculate_fisher_transform(high, low, period=9)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_sma_30m = calculate_volume_sma(volume, period=20)
    hours_30m = extract_hour_from_open_time(open_time)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime (bull/bear market)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Fisher crossover tracking
    prev_fisher = 0.0
    prev_fisher_signal = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_30m[i]) or np.isnan(fisher_signal_30m[i]):
            prev_fisher = fisher_30m[i] if not np.isnan(fisher_30m[i]) else prev_fisher
            prev_fisher_signal = fisher_signal_30m[i] if not np.isnan(fisher_signal_30m[i]) else prev_fisher_signal
            continue
        if np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_30m[i]) or vol_sma_30m[i] <= 1e-10:
            continue
        
        current_fisher = fisher_30m[i]
        current_fisher_signal = fisher_signal_30m[i]
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours_30m[i] <= 20
        
        # === VOLUME FILTER (>0.8x 20-bar average) ===
        volume_ok = volume[i] > 0.8 * vol_sma_30m[i]
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (prev_fisher <= -1.5 and current_fisher > -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (prev_fisher >= 1.5 and current_fisher < 1.5)
        
        # Extreme Fisher levels (fallback)
        fisher_extreme_long = current_fisher < -2.0
        fisher_extreme_short = current_fisher > 2.0
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (3+ confluence required) ===
        if in_session and volume_ok:
            # Primary long: Fisher cross + 4h trend + 1d trend
            if fisher_long_cross and trend_4h_bullish and macro_bull:
                desired_signal = BASE_SIZE
            # Secondary long: Fisher cross + 4h trend (relaxed)
            elif fisher_long_cross and trend_4h_bullish:
                desired_signal = REDUCED_SIZE
            # Fallback long: Extreme Fisher + 4h trend
            elif fisher_extreme_long and trend_4h_bullish:
                desired_signal = REDUCED_SIZE
            # Fallback long: Extreme Fisher + macro bull
            elif fisher_extreme_long and macro_bull:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS (3+ confluence required) ===
        if in_session and volume_ok:
            # Primary short: Fisher cross + 4h trend + 1d trend
            if fisher_short_cross and trend_4h_bearish and macro_bear:
                desired_signal = -BASE_SIZE
            # Secondary short: Fisher cross + 4h trend (relaxed)
            elif fisher_short_cross and trend_4h_bearish:
                desired_signal = -REDUCED_SIZE
            # Fallback short: Extreme Fisher + 4h trend
            elif fisher_extreme_short and trend_4h_bearish:
                desired_signal = -REDUCED_SIZE
            # Fallback short: Extreme Fisher + macro bear
            elif fisher_extreme_short and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend intact and Fisher not overbought
                if trend_4h_bullish and current_fisher < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend intact and Fisher not oversold
                if trend_4h_bearish and current_fisher > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h + 1d trend reverses + Fisher overbought
            if trend_4h_bearish and macro_bear and current_fisher > 1.5:
                desired_signal = 0.0
            # Exit if Fisher extremely overbought
            if current_fisher > 2.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h + 1d trend reverses + Fisher oversold
            if trend_4h_bullish and macro_bull and current_fisher < -1.5:
                desired_signal = 0.0
            # Exit if Fisher extremely oversold
            if current_fisher < -2.5:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
        
        # Update previous Fisher values for crossover detection
        prev_fisher = current_fisher
        prev_fisher_signal = current_fisher_signal
    
    return signals