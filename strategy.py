#!/usr/bin/env python3
"""
Experiment #1510: 1h Primary + 4h/12h HTF — Fisher Transform + HMA Trend

Hypothesis: After analyzing 1100+ failed strategies, the pattern is clear:
1. Complex filters (CHOP+CRSI+session+volume) = 0 trades (#1498, #1500, #1508)
2. Simpler HMA+RSI works (#1505, #1506 kept) but RSI lags in bear markets
3. FISHER TRANSFORM is superior for reversals in bear/range markets (2022, 2025)
4. 1h needs 4h/12h for DIRECTION, 1h only for ENTRY TIMING
5. Loose entry conditions are MANDATORY for trades (Fisher -1.5/+1.5, not extremes)

Key design choices:
- 12h HMA(21) for macro regime (bull/bear bias)
- 4h HMA(21) for intermediate trend direction
- 1h Fisher Transform(9) for precise entry timing (reversals)
- 1h Volume filter (loose: >0.6x avg, not 0.8x) to ensure trades
- ATR(14) 2.5x trailing stop for risk management
- Position size 0.25 (appropriate for 1h trade frequency ~40-60/year)
- Discrete signal levels (0.0, ±0.25) to minimize fee churn

Timeframe: 1h (as required by experiment)
HTF: 4h (trend), 12h (regime)
Position Size: 0.25 (discrete: 0.0, ±0.25)
Target: 120-240 trades/train (4 years), 30-60 trades/test (15 months), Sharpe > 0.618
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_4h12h_regime_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Excellent for catching reversals in bear/range markets
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.33 * 2 * (close - LL) / (HH - LL) - 0.33
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    # Calculate price range normalization
    for i in range(period - 1, n):
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh - ll < 1e-10:
            continue
        
        # Normalize price to -1 to +1 range
        x = 0.33 * 2.0 * ((close[i] - ll) / (hh - ll) - 0.5)
        x = np.clip(x, -0.999, 0.999)  # Prevent division by zero in ln
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Previous Fisher value (for crossover detection)
        if i > period - 1:
            fisher_prev[i] = fisher[i - 1]
    
    return fisher, fisher_prev

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

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
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
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for regime bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Appropriate size for 1h (moderate trade frequency)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME BIAS (12h HMA) - macro direction ===
        regime_bull = close[i] > hma_12h_aligned[i]
        regime_bear = close[i] < hma_12h_aligned[i]
        
        # === TREND DIRECTION (4h HMA) - intermediate confirmation ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === VOLUME FILTER (loose to ensure trades) ===
        vol_confirmed = volume[i] > 0.6 * vol_sma[i]
        
        # === FISHER TRANSFORM ENTRY - LOOSE thresholds ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = fisher[i] > -1.5 and fisher_prev[i] <= -1.5
        # Long: Fisher already above -1.0 and rising (momentum continuation)
        fisher_long_cont = fisher[i] > -1.0 and fisher[i] > fisher_prev[i]
        
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = fisher[i] < 1.5 and fisher_prev[i] >= 1.5
        # Short: Fisher already below +1.0 and falling (momentum continuation)
        fisher_short_cont = fisher[i] < 1.0 and fisher[i] < fisher_prev[i]
        
        # === DESIRED SIGNAL - LOOSE conditions for TRADES ===
        desired_signal = 0.0
        
        # LONG scenarios (multiple paths to ensure trades happen)
        # Path 1: Strong regime + trend + Fisher reversal + volume
        if regime_bull and trend_bull and fisher_long and vol_confirmed:
            desired_signal = BASE_SIZE
        # Path 2: Regime bull + trend bull + Fisher continuation (looser)
        elif regime_bull and trend_bull and fisher_long_cont:
            desired_signal = BASE_SIZE * 0.8
        # Path 3: Regime bull + Fisher reversal (fallback for trades)
        elif regime_bull and fisher_long:
            desired_signal = BASE_SIZE * 0.6
        # Path 4: Trend bull + Fisher not overbought (loosest)
        elif trend_bull and fisher[i] < 1.0:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT scenarios
        # Path 1: Strong regime + trend + Fisher reversal + volume
        elif regime_bear and trend_bear and fisher_short and vol_confirmed:
            desired_signal = -BASE_SIZE
        # Path 2: Regime bear + trend bear + Fisher continuation (looser)
        elif regime_bear and trend_bear and fisher_short_cont:
            desired_signal = -BASE_SIZE * 0.8
        # Path 3: Regime bear + Fisher reversal (fallback for trades)
        elif regime_bear and fisher_short:
            desired_signal = -BASE_SIZE * 0.6
        # Path 4: Trend bear + Fisher not oversold (loosest)
        elif trend_bear and fisher[i] > -1.0:
            desired_signal = -BASE_SIZE * 0.5
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.6
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals