#!/usr/bin/env python3
"""
Experiment #875: 1h Primary + 4h/1d HTF — Fisher Transform + Choppiness + Session

Hypothesis: After 600+ failed strategies, 1h timeframe with Fisher Transform entries
should work better than CRSI/RSI for catching reversals in bear/range markets.
Key insights from research:

1. 1h Primary TF: Target 30-60 trades/year (use strict confluence filters)
2. 4h HMA(21) for trend direction (HTF bias - only trade with HTF trend)
3. 1d HMA(21) for macro regime filter (bull/bear market)
4. Ehlers Fisher Transform(9) for entry timing - catches reversals better than RSI
5. Choppiness Index(14) for regime: CHOP>55=range (mean revert), CHOP<45=trend (breakout)
6. Session filter: only trade 8-20 UTC (high volume hours, reduces false signals)
7. Volume confirmation: volume > 0.8x 20-bar average
8. ATR(14) trailing stop (2.5x) for risk management

Why Fisher Transform over RSI/CRSI:
- Fisher normalizes price to Gaussian distribution (-1 to +1)
- Better at identifying extreme reversals in bear markets
- Less lag than RSI, more responsive to price changes
- Proven in research for crypto mean reversion

Why this should work on 1h:
- HTF (4h/1d) provides strong trend bias
- Session filter reduces trades during low-volume hours (fewer false signals)
- Fisher + Choppiness combo catches both trend and range markets
- Discrete signal sizes (0.0, ±0.20, ±0.30) minimize fee churn
- Volume filter ensures we only trade when there's actual participation

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_regime_4h1d_hma_session_vol_atr_v1"
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
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest)
    3. Fisher: 0.5 * ln((1 + x) / (1 - x)) where x = 2 * normalized - 1
    
    Long when Fisher crosses above -1.5 (oversold reversal)
    Short when Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    for i in range(period - 1, n):
        # Get highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
            fisher_prev[i] = fisher[i-1] if i > 1 and not np.isnan(fisher[i-1]) else 0.0
            continue
        
        # Typical price
        typical = (high[i] + low[i]) / 2.0
        
        # Normalize to 0-1 range
        normalized = (typical - lowest) / (highest - lowest)
        
        # Transform to -1 to +1 range, then apply Fisher
        x = 2.0 * normalized - 1.0
        
        # Clamp to avoid division by zero
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Previous value for crossover detection
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

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
    fisher_1h, fisher_prev_1h = calculate_fisher_transform(high, low, period=9)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(fisher_1h[i]) or np.isnan(fisher_prev_1h[i]):
            continue
        if np.isnan(chop_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_1h[i]) or vol_sma_1h[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        volume_confirmed = volume[i] > 0.8 * vol_sma_1h[i]
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (1h Choppiness Index) ===
        ranging_regime = chop_1h[i] > 55
        trending_regime = chop_1h[i] < 45
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher_1h[i] < -1.5
        fisher_overbought = fisher_1h[i] > 1.5
        fisher_cross_up = fisher_prev_1h[i] < -1.0 and fisher_1h[i] >= -1.0
        fisher_cross_down = fisher_prev_1h[i] > 1.0 and fisher_1h[i] <= 1.0
        fisher_extreme_oversold = fisher_1h[i] < -2.0
        fisher_extreme_overbought = fisher_1h[i] > 2.0
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion with Fisher ===
        if ranging_regime:
            # Long: Fisher oversold + HTF trend alignment + session + volume
            if fisher_oversold and (macro_bull or trend_4h_bullish) and in_session and volume_confirmed:
                desired_signal = BASE_SIZE
            
            # Short: Fisher overbought + HTF trend alignment + session + volume
            if fisher_overbought and (macro_bear or trend_4h_bearish) and in_session and volume_confirmed:
                desired_signal = -BASE_SIZE
            
            # Fallback: extreme Fisher alone (guarantees trades in ranging market)
            if fisher_extreme_oversold and desired_signal == 0 and in_session:
                desired_signal = REDUCED_SIZE
            
            if fisher_extreme_overbought and desired_signal == 0 and in_session:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish HTF trend + Fisher crossing up from oversold
            if (macro_bull or trend_4h_bullish) and fisher_cross_up and in_session and volume_confirmed:
                desired_signal = BASE_SIZE
            
            # Short: Bearish HTF trend + Fisher crossing down from overbought
            if (macro_bear or trend_4h_bearish) and fisher_cross_down and in_session and volume_confirmed:
                desired_signal = -BASE_SIZE
            
            # Fallback: trend + moderate Fisher signal
            if (macro_bull or trend_4h_bullish) and fisher_oversold and in_session:
                desired_signal = REDUCED_SIZE
            
            if (macro_bear or trend_4h_bearish) and fisher_overbought and in_session:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Fisher extreme + HTF confluence + session
            if fisher_extreme_oversold and (macro_bull or trend_4h_bullish) and in_session:
                desired_signal = REDUCED_SIZE
            
            if fisher_extreme_overbought and (macro_bear or trend_4h_bearish) and in_session:
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
            # Exit long if HTF trend reverses + Fisher overbought
            if macro_bear and trend_4h_bearish and fisher_1h[i] > 1.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HTF trend reverses + Fisher oversold
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