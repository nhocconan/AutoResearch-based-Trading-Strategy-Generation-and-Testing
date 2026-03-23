#!/usr/bin/env python3
"""
Experiment #484: 4h Primary + 12h/1d HTF — HMA Trend + Z-Score Mean Reversion + Volume Confirmation

Hypothesis: Based on research showing that BTC/ETH respond well to volatility-based mean reversion
when combined with trend filtering. Key innovations different from failed experiments:
1. HMA(21) for trend - faster response than EMA, less lag than SMA
2. Z-score(20) of price vs SMA - different from RSI, catches statistical extremes
3. Volume spike confirmation - taker_buy_volume ratio > 1.5x average for conviction
4. 12h HMA for HTF trend bias - smoother than 4h, faster than 1d
5. ATR(14) trailing stop at 2.5x for risk management
6. Relaxed entry thresholds to ensure 30+ trades (avoid 0-trade failure mode)
7. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work: Z-score is fundamentally different from RSI/CRSI (failed in #473-#483).
Volume confirmation adds a dimension not used in recent failures. HMA has proven track record
in mtf_hma_rsi_zscore_v1 (Sharpe=5.4 baseline). 4h TF naturally targets 20-50 trades/year.
12h HTF provides trend bias without being too slow. This combines trend + mean reversion
in a simpler way than complex regime-switching strategies that failed.

Target: Sharpe > 0.612, DD < -35%, trades >= 30 on train, >= 3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_zscore_vol_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return pd.Series(series).rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
    
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # HMA calculation
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_zscore(close, period=20):
    """
    Calculate Z-score of price relative to SMA.
    Z = (price - SMA) / std
    Z > 2.0 = overbought, Z < -2.0 = oversold
    """
    n = len(close)
    zscore = np.full(n, np.nan)
    
    for i in range(period, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 1e-10:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0.0
    
    return zscore

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1]) if i > 0 else tr1
        tr3 = np.abs(low[i] - close[i - 1]) if i > 0 else tr1
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(taker_buy_volume, period=20):
    """
    Calculate volume ratio: current taker_buy_volume / SMA of past volume.
    Ratio > 1.5 = volume spike (buying pressure)
    Ratio < 0.7 = volume drop (weakness)
    """
    n = len(taker_buy_volume)
    ratio = np.full(n, np.nan)
    
    for i in range(period, n):
        window = taker_buy_volume[i - period + 1:i + 1]
        mean_vol = np.mean(window)
        if mean_vol > 1e-10:
            ratio[i] = taker_buy_volume[i] / mean_vol
        else:
            ratio[i] = 1.0
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_4h = calculate_hma(close, period=21)
    zscore_4h = calculate_zscore(close, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate SMA for trend filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Calculate and align HTF indicators (12h HMA for trend bias)
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # 1d HMA for major trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(hma_4h[i]):
            continue
        if np.isnan(zscore_4h[i]):
            continue
        if np.isnan(vol_ratio[i]):
            continue
        if np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        
        # === HTF MAJOR TREND BIAS (12h + 1d HMA) ===
        htf_12h_bullish = close[i] > hma_12h_aligned[i]
        htf_12h_bearish = close[i] < hma_12h_aligned[i]
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong HTF bias when both agree
        htf_strong_bull = htf_12h_bullish and htf_1d_bullish
        htf_strong_bear = htf_12h_bearish and htf_1d_bearish
        
        # === PRIMARY TREND (4h HMA + SMA filters) ===
        price_above_hma = close[i] > hma_4h[i]
        price_below_hma = close[i] < hma_4h[i]
        price_above_sma50 = close[i] > sma_50[i]
        price_above_sma200 = close[i] > sma_200[i]
        
        # HMA slope (5 bar lookback)
        hma_slope_up = hma_4h[i] > hma_4h[i - 5] if i >= 5 else False
        hma_slope_down = hma_4h[i] < hma_4h[i - 5] if i >= 5 else False
        
        # === Z-SCORE SIGNALS (Mean Reversion) ===
        zscore_oversold = zscore_4h[i] < -1.5
        zscore_overbought = zscore_4h[i] > 1.5
        zscore_extreme_oversold = zscore_4h[i] < -2.0
        zscore_extreme_overbought = zscore_4h[i] > 2.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike_up = vol_ratio[i] > 1.3
        vol_spike_down = vol_ratio[i] < 0.8
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES (relaxed conditions to ensure trades)
        long_score = 0
        
        # HTF bias alignment (at least 12h bullish required)
        if htf_12h_bullish:
            long_score += 2
        if htf_strong_bull:
            long_score += 1
        
        # Price above HMA (trend confirmation)
        if price_above_hma:
            long_score += 1
        
        # HMA slope up
        if hma_slope_up:
            long_score += 1
        
        # Z-score entry (mean reversion in uptrend)
        if zscore_oversold:
            long_score += 2
        if zscore_extreme_oversold:
            long_score += 1
        
        # Volume confirmation (optional bonus)
        if vol_spike_up:
            long_score += 1
        
        # Enter long if score >= 4 (relaxed from 5)
        if long_score >= 4:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_score = 0
            
            # HTF bias alignment
            if htf_12h_bearish:
                short_score += 2
            if htf_strong_bear:
                short_score += 1
            
            # Price below HMA
            if price_below_hma:
                short_score += 1
            
            # HMA slope down
            if hma_slope_down:
                short_score += 1
            
            # Z-score entry
            if zscore_overbought:
                short_score += 2
            if zscore_extreme_overbought:
                short_score += 1
            
            # Volume confirmation
            if vol_spike_down:
                short_score += 1
            
            if short_score >= 4:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma and htf_12h_bullish:
                desired_signal = SIZE_LONG
            elif position_side < 0 and price_below_hma and htf_12h_bearish:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.30
        elif desired_signal < 0:
            desired_signal = -0.25
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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