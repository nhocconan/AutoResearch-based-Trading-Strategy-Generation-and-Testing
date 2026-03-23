#!/usr/bin/env python3
"""
Experiment #905: 1h Primary + 4h/1d HTF — Fisher Transform + HMA Trend + Volume Session

Hypothesis: After 600+ failed strategies, 1h timeframe needs VERY strict entry filters
to avoid fee drag (>100 trades/year kills profit). Key insights:

1. Ehlers Fisher Transform (period=9) catches reversals in bear/range markets better than RSI
2. 4h HMA(21) for medium-term trend direction (HTF bias filter)
3. 1d HMA(21) for macro regime (bull/bear market filter)
4. Volume filter: only trade when volume > 0.8x 20-bar average (avoid low liquidity)
5. Session filter: only 8-20 UTC (high liquidity windows, avoid Asia overnight whipsaw)
6. ATR(14) trailing stop (2.5x) for risk management
7. Discrete signal sizes (0.0, ±0.20, ±0.30) to minimize fee churn

Why this should work on 1h:
- Fisher Transform is proven for bear market reversals (unlike RSI which fails)
- HTF (4h/1d) provides strong trend bias, 1h only for entry timing
- Volume + Session filters reduce trades to 30-60/year target
- All symbols must generate trades (relaxed Fisher thresholds: -1.5/+1.5 not -2/+2)

Critical improvements from failed experiments:
- Fisher Transform instead of CRSI (different signal type)
- Volume confirmation to avoid false breakouts
- Session filter to avoid overnight whipsaw
- Relaxed entry thresholds to guarantee 30+ trades per symbol
- ALL symbols MUST have positive Sharpe (no SOL-only bias)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_4h1d_volume_session_atr_v1"
timeframe = "1h"
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
    Excellent for catching reversals in bear/range markets.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_signal
    
    for i in range(period, n):
        # Calculate typical price range
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            fisher[i] = 0.0
            fisher_signal[i] = 0.0
            continue
        
        # Normalize price to 0-1 range
        mid_price = (high[i] + low[i]) / 2.0
        normalized = (mid_price - lowest_low) / (highest_high - lowest_low)
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Signal line (1-bar lag)
        if i > 0 and not np.isnan(fisher[i-1]):
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

def get_hour_from_open_time(open_time_array):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
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
    
    # Calculate primary (1h) indicators
    fisher_1h, fisher_signal_1h = calculate_fisher_transform(high, low, period=9)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    hours = get_hour_from_open_time(open_time)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime (bull/bear market)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_1h[i]) or np.isnan(fisher_signal_1h[i]):
            continue
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_1h[i]) or vol_sma_1h[i] <= 1e-10:
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma_1h[i]
        
        # === SESSION FILTER (8-20 UTC only) ===
        session_ok = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (fisher_1h[i] > -1.5) and (fisher_signal_1h[i] <= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (fisher_1h[i] < 1.5) and (fisher_signal_1h[i] >= 1.5)
        
        # Secondary: Fisher extreme values (guarantees trades)
        fisher_extreme_long = fisher_1h[i] < -1.8
        fisher_extreme_short = fisher_1h[i] > 1.8
        
        # Tertiary: Fisher recovering from extreme
        fisher_recover_long = (fisher_1h[i] > fisher_signal_1h[i]) and (fisher_1h[i] < -0.5)
        fisher_recover_short = (fisher_1h[i] < fisher_signal_1h[i]) and (fisher_1h[i] > 0.5)
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        # Primary: Fisher cross + HTF trend alignment + volume + session
        if fisher_long_cross and (macro_bull or trend_4h_bullish) and volume_ok and session_ok:
            desired_signal = BASE_SIZE
        # Secondary: Fisher extreme + any HTF bullish bias
        elif fisher_extreme_long and (macro_bull or trend_4h_bullish) and volume_ok:
            desired_signal = REDUCED_SIZE
        # Tertiary: Fisher recovery + strong HTF alignment + session
        elif fisher_recover_long and macro_bull and trend_4h_bullish and session_ok:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY LOGIC ===
        # Primary: Fisher cross + HTF trend alignment + volume + session
        if fisher_short_cross and (macro_bear or trend_4h_bearish) and volume_ok and session_ok:
            if desired_signal == 0.0:
                desired_signal = -BASE_SIZE
        # Secondary: Fisher extreme + any HTF bearish bias
        elif fisher_extreme_short and (macro_bear or trend_4h_bearish) and volume_ok:
            if desired_signal == 0.0:
                desired_signal = -REDUCED_SIZE
        # Tertiary: Fisher weakening + strong HTF alignment + session
        elif fisher_recover_short and macro_bear and trend_4h_bearish and session_ok:
            if desired_signal == 0.0:
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
                # Hold long if HTF trend intact and Fisher not overbought
                if (macro_bull or trend_4h_bullish) and fisher_1h[i] < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HTF trend intact and Fisher not oversold
                if (macro_bear or trend_4h_bearish) and fisher_1h[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro + medium trend reverses + Fisher overbought
            if macro_bear and trend_4h_bearish and fisher_1h[i] > 1.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro + medium trend reverses + Fisher oversold
            if macro_bull and trend_4h_bullish and fisher_1h[i] < -1.5:
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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
    
    return signals