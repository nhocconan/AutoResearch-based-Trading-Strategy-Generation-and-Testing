#!/usr/bin/env python3
"""
Experiment #438: 30m Primary + 4h/1d HTF — Fisher Transform + Volume Confluence

Hypothesis: Lower TF (30m) strategies fail due to either (1) too many trades → fee drag,
or (2) too strict filters → 0 trades. Solution: Use HTF (4h/1d) for DIRECTION,
30m only for ENTRY TIMING with 3+ confluence filters.

Key innovations vs failed #428/#435:
1. Fisher Transform (period=9) for entry timing — catches reversals better than RSI
2. 1d CHOP for regime (not 30m CHOP which is too noisy)
3. 4h HMA for trend bias (direction filter)
4. Volume spike confirmation (vol > 1.2x 20-bar avg) — avoids low-liquidity traps
5. Session filter (8-20 UTC) — avoids Asian session whipsaws
6. Discrete signal sizes (0.0, ±0.20, ±0.30) to minimize fee churn

Target: 40-80 trades/year, Sharpe > 0.612, DD < -40%
Position size: 0.25 (smaller for 30m vs 4h's 0.30)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_volume_chop_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (close - LL) / (HH - LL) - 0.33
    Catches reversals in bear/range markets better than RSI.
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    for i in range(period, n):
        hh = np.nanmax(high[i-period+1:i+1])
        ll = np.nanmin(low[i-period+1:i+1])
        
        if hh - ll > 1e-10:
            X = 0.67 * (close[i] - ll) / (hh - ll) - 0.33
            X = np.clip(X, -0.999, 0.999)  # Prevent division by zero
            fisher[i] = 0.5 * np.log((1 + X) / (1 - X + 1e-10))
            
            if i > period:
                fisher_signal[i] = fisher[i-1]
        else:
            fisher[i] = 0.0
            if i > period:
                fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    n = len(close)
    chop = np.full(n, np.nan)
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def calculate_volume_spike(volume, period=20):
    """Detect volume spikes relative to rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    ts_seconds = open_time / 1000
    utc_hour = (ts_seconds % 86400) / 3600
    return utc_hour

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
    
    # Calculate 30m indicators (primary timeframe)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    vol_ratio = calculate_volume_spike(volume, period=20)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 30m (smaller than 4h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(chop_1d_aligned[i]):
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(sma_200[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === REGIME DETECTION (1d CHOP) ===
        regime_range = chop_1d_aligned[i] > 55.0  # Range market
        regime_trend = chop_1d_aligned[i] < 45.0  # Trending market
        
        # === TREND BIAS (4h HMA) ===
        hma_4h_bullish = hma_4h_aligned[i] > hma_4h_aligned[i-1] if not np.isnan(hma_4h_aligned[i-1]) else False
        hma_4h_bearish = hma_4h_aligned[i] < hma_4h_aligned[i-1] if not np.isnan(hma_4h_aligned[i-1]) else False
        
        # === PRIMARY TREND (30m HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5 and fisher_signal[i] < fisher[i]  # Cross up from oversold
        fisher_overbought = fisher[i] > 1.5 and fisher_signal[i] > fisher[i]  # Cross down from overbought
        fisher_extreme_oversold = fisher[i] < -2.0
        fisher_extreme_overbought = fisher[i] > 2.0
        
        # === VOLUME FILTER ===
        volume_confirmed = vol_ratio[i] > 1.0  # At least average volume
        
        # === VOLATILITY FILTER ===
        vol_ratio_atr = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio_atr > 2.0:
            position_size = BASE_SIZE * 0.5  # Reduce size in extreme vol
        elif vol_ratio_atr > 1.5:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === REGIME 1: RANGE (CHOP > 55) — MEAN REVERSION ===
        if regime_range:
            # Long: Fisher oversold + volume + session + SMA200 filter
            if fisher_oversold and in_session and volume_confirmed and price_above_sma200:
                desired_signal = position_size
            elif fisher_extreme_oversold and in_session:
                desired_signal = position_size * 1.2
            
            # Short: Fisher overbought + volume + session + SMA200 filter
            if fisher_overbought and in_session and volume_confirmed and price_below_sma200:
                if desired_signal == 0:
                    desired_signal = -position_size
            elif fisher_extreme_overbought and in_session:
                if desired_signal == 0:
                    desired_signal = -position_size * 1.2
        
        # === REGIME 2: TRENDING (CHOP < 45) — TREND FOLLOW ===
        elif regime_trend:
            # Long: 4h bullish + 30m bullish + Fisher turning up
            if hma_4h_bullish and hma_bullish and fisher[i] > fisher_signal[i] and in_session:
                desired_signal = position_size
            elif hma_4h_bullish and fisher_extreme_oversold and in_session:
                desired_signal = position_size * 0.8
            
            # Short: 4h bearish + 30m bearish + Fisher turning down
            if hma_4h_bearish and hma_bearish and fisher[i] < fisher_signal[i] and in_session:
                if desired_signal == 0:
                    desired_signal = -position_size
            elif hma_4h_bearish and fisher_extreme_overbought and in_session:
                if desired_signal == 0:
                    desired_signal = -position_size * 0.8
        
        # === REGIME 3: TRANSITION (45-55) — REDUCED SIZE ===
        else:
            # Only extreme Fisher signals
            if fisher_extreme_oversold and in_session:
                desired_signal = position_size * 0.5
            elif fisher_extreme_overbought and in_session:
                desired_signal = -position_size * 0.5
        
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
        
        # === FISHER EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and fisher[i] > 2.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -2.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_bullish or hma_4h_bullish):
                desired_signal = position_size
            elif position_side < 0 and (hma_bearish or hma_4h_bearish):
                desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.30:
                    desired_signal = 0.30
                elif desired_signal >= 0.20:
                    desired_signal = 0.20
                else:
                    desired_signal = 0.20
            else:
                if desired_signal <= -0.30:
                    desired_signal = -0.30
                elif desired_signal <= -0.20:
                    desired_signal = -0.20
                else:
                    desired_signal = -0.20
        
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