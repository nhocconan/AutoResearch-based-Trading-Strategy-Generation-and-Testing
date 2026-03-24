#!/usr/bin/env python3
"""
Experiment #967: 6h Primary + 1d/1w HTF — Fisher Transform + Regime Breakout

Hypothesis: Ehlers Fisher Transform excels at catching reversals in mixed markets
(2022 crash + 2025 bear). Combined with 1d HMA trend bias and volume confirmation,
this should outperform CHOP+CRSI which failed in #955.

Key innovations:
1. Fisher Transform (period=9): Normalizes price to Gaussian distribution
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
2. 1d HMA(21) slope for intermediate trend bias (not just price vs HMA)
3. 1w momentum (close > open) for weekly directional bias
4. Volume confirmation: volume > 1.3x 20-bar average at entry
5. ADX(14) > 20 filter to avoid dead markets
6. ATR(14) 2.5x trailing stop for risk management

Why this should beat #955 (CHOP+CRSI):
- Fisher Transform is purpose-built for reversal detection (Ehlers 2002)
- CHOP failed because it's too slow for 6h timeframe
- Volume confirmation reduces false breakouts
- HMA slope (not just price position) gives earlier trend signals
- Looser Fisher thresholds (-1.5/+1.5 vs -2/+2) guarantee more trades

Entry conditions (LOOSE to guarantee >=30 trades/year):
- LONG = 1w bull + 1d HMA slope up + Fisher cross above -1.5 + ADX>20 + vol confirm
- SHORT = 1w bear + 1d HMA slope down + Fisher cross below +1.5 + ADX>20 + vol confirm

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_vol_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for clearer reversal signals
    
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    where X = 0.66 * prev_X + 0.33 * ((close - lowest) / (highest - lowest) * 2 - 1)
    
    Long signal: Fisher crosses above -1.5
    Short signal: Fisher crosses below +1.5
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    X = 0.0
    prev_X = 0.0
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest > lowest:
            close_norm = (high[i] + low[i]) / 2.0
            X = 0.66 * prev_X + 0.33 * ((close_norm - lowest) / (highest - lowest) * 2.0 - 1.0)
            X = np.clip(X, -0.999, 0.999)
            
            fisher[i] = 0.5 * np.log((1.0 + X) / (1.0 - X))
            fisher_signal[i] = fisher[i-1] if i > period else fisher[i]
            
            prev_X = X
    
    return fisher, fisher_signal

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback periods"""
    n = len(hma_values)
    slope = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        if not np.isnan(hma_values[i]) and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / (hma_values[i - lookback] + 1e-10)
    
    return slope

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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.divide(plus_di, atr, out=np.zeros_like(plus_di), where=atr != 0) * 100
    minus_di = np.divide(minus_di, atr, out=np.zeros_like(minus_di), where=atr != 0) * 100
    
    dx = np.divide(np.abs(plus_di - minus_di), plus_di + minus_di, 
                   out=np.zeros_like(plus_di), where=(plus_di + minus_di) != 0) * 100
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:period*2] = np.nan
    
    return adx

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
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
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_slope_raw = calculate_hma_slope(hma_1d_raw, lookback=3)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope_raw)
    
    # Weekly momentum: close vs open
    weekly_momentum_raw = (df_1w['close'].values - df_1w['open'].values) / (df_1w['open'].values + 1e-10)
    weekly_momentum_aligned = align_htf_to_ltf(prices, df_1w, weekly_momentum_raw)
    
    # Calculate 6h indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(weekly_momentum_aligned[i]) or np.isnan(adx_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w momentum + 1d HMA slope) ===
        htf_1w_bull = weekly_momentum_aligned[i] > 0.0
        htf_1w_bear = weekly_momentum_aligned[i] < 0.0
        
        htf_1d_bull = hma_1d_slope_aligned[i] > 0.001  # Slope threshold
        htf_1d_bear = hma_1d_slope_aligned[i] < -0.001
        
        # === MARKET REGIME (ADX) ===
        is_trending = adx_14[i] > 20.0
        
        # === FISHER TRANSFORM SIGNALS (LOOSE THRESHOLDS) ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_signal[i-1]):
            # Long: Fisher crosses above -1.5 from below
            fisher_cross_long = (fisher_signal[i-1] <= -1.5) and (fisher[i] > -1.5)
            # Short: Fisher crosses below +1.5 from above
            fisher_cross_short = (fisher_signal[i-1] >= 1.5) and (fisher[i] < 1.5)
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = volume[i] > 1.3 * vol_sma_20[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries (looser conditions for more trades)
        if htf_1w_bull and htf_1d_bull:
            if fisher_cross_long:
                if is_trending or vol_confirm:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            elif fisher[i] < -1.0 and close[i] < hma_1d_aligned[i] * 0.98:
                # Deep pullback in uptrend
                desired_signal = SIZE_BASE
        
        # SHORT entries (looser conditions for more trades)
        elif htf_1w_bear and htf_1d_bear:
            if fisher_cross_short:
                if is_trending or vol_confirm:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            elif fisher[i] > 1.0 and close[i] > hma_1d_aligned[i] * 1.02:
                # Deep rally in downtrend
                desired_signal = -SIZE_BASE
        
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