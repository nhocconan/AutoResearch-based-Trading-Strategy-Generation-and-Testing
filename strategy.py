#!/usr/bin/env python3
"""
Experiment #423: 1d Primary + 1w HTF — Fisher Transform Reversals + HMA Trend

Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets
(2022 crash, 2025 bear). Combined with 1w HMA for trend bias, this should produce
30-60 trades over 4 years with better risk-adjusted returns than regime-switching.

Key innovations vs #417:
1. Fisher Transform instead of CRSI — better at catching sharp reversals
2. Simpler entry logic — fewer nested conditions = more trades
3. 1w HMA slope for trend confirmation (not just price position)
4. More permissive entry thresholds to ensure trade frequency
5. ATR(14) trailing stoploss at 2.5x for risk management

Why this should beat #417 (Sharpe=0.042):
- Fisher Transform normalizes price to Gaussian distribution, better signal-to-noise
- Weekly HMA slope is stronger trend filter than price position alone
- Simpler logic = fewer conditions that can all fail simultaneously
- Tested on 1d which has proven more stable than 4h/12h in recent experiments

Target: Sharpe > 0.5, 80-200 trades over 4-year train, DD < -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_hma_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (close - lowest) / (highest - lowest) - 0.67
    Signals: Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    for i in range(period, n):
        highest = close[i-period+1:i+1].max()
        lowest = close[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
            fisher_signal[i] = fisher_signal[i-1] if i > 0 and not np.isnan(fisher_signal[i-1]) else 0.0
            continue
        
        X = 0.67 * (close[i] - lowest) / (highest - lowest) - 0.67
        X = np.clip(X, -0.999, 0.999)  # Prevent log domain errors
        
        fisher[i] = 0.5 * np.log((1.0 + X) / (1.0 - X + 1e-10))
        
        # Smooth Fisher with 1-period EMA for signal line
        if i > period and not np.isnan(fisher[i-1]):
            fisher_signal[i] = 0.5 * fisher[i] + 0.5 * fisher_signal[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

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

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_hma_slope(hma, lookback=5):
    """Calculate HMA slope (rate of change over lookback periods)."""
    n = len(hma)
    slope = np.full(n, np.nan)
    for i in range(lookback, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i-lookback]):
            slope[i] = (hma[i] - hma[i-lookback]) / (hma[i-lookback] + 1e-10)
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    fisher, fisher_signal = calculate_fisher(close, period=9)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    hma_21_slope = calculate_hma_slope(hma_21, lookback=5)
    atr_14 = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate and align HTF HMA for bias (1w)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    hma_1w_slope = calculate_hma_slope(hma_1w_aligned, lookback=3)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 1d
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crossings for entry timing
    prev_fisher_signal = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            prev_fisher_signal = fisher_signal[i] if not np.isnan(fisher_signal[i]) else prev_fisher_signal
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            prev_fisher_signal = fisher_signal[i] if not np.isnan(fisher_signal[i]) else prev_fisher_signal
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            prev_fisher_signal = fisher_signal[i] if not np.isnan(fisher_signal[i]) else prev_fisher_signal
            continue
        if np.isnan(sma_200[i]) or np.isnan(rsi_14[i]):
            prev_fisher_signal = fisher_signal[i] if not np.isnan(fisher_signal[i]) else prev_fisher_signal
            continue
        
        # === HTF BIAS (1w HMA slope + price position) ===
        hma_1w_bullish = hma_1w_slope[i] > 0.001 and close[i] > hma_1w_aligned[i]
        hma_1w_bearish = hma_1w_slope[i] < -0.001 and close[i] < hma_1w_aligned[i]
        hma_1w_neutral = abs(hma_1w_slope[i]) <= 0.001
        
        # === PRIMARY TREND (1d HMA) ===
        hma_21_bullish = hma_21[i] > hma_50[i] and hma_21_slope[i] > 0
        hma_21_bearish = hma_21[i] < hma_50[i] and hma_21_slope[i] < 0
        
        # === SMA200 FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = fisher_signal[i] > -1.5 and prev_fisher_signal <= -1.5
        fisher_cross_short = fisher_signal[i] < 1.5 and prev_fisher_signal >= 1.5
        fisher_extreme_long = fisher[i] < -2.0  # Deep oversold
        fisher_extreme_short = fisher[i] > 2.0  # Deep overbought
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.8:
            position_size = BASE_SIZE * 0.7
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP — Multiple confluence paths (any can trigger)
        long_bias = hma_1w_bullish or (hma_1w_neutral and price_above_sma200)
        
        if long_bias:
            # Path 1: Fisher cross + HMA bullish
            if fisher_cross_long and hma_21_bullish:
                desired_signal = position_size
            # Path 2: Fisher extreme + RSI oversold (mean reversion)
            elif fisher_extreme_long and rsi_oversold:
                desired_signal = position_size
            # Path 3: HMA bullish pullback (RSI moderate)
            elif hma_21_bullish and rsi_14[i] < 50.0 and fisher_signal[i] < 0:
                desired_signal = position_size * 0.7
        
        # SHORT SETUP — Multiple confluence paths (any can trigger)
        short_bias = hma_1w_bearish or (hma_1w_neutral and price_below_sma200)
        
        if short_bias:
            # Path 1: Fisher cross + HMA bearish
            if fisher_cross_short and hma_21_bearish:
                desired_signal = -position_size
            # Path 2: Fisher extreme + RSI overbought (mean reversion)
            elif fisher_extreme_short and rsi_overbought:
                desired_signal = -position_size
            # Path 3: HMA bearish rally (RSI moderate)
            elif hma_21_bearish and rsi_14[i] > 50.0 and fisher_signal[i] > 0:
                desired_signal = -position_size * 0.7
        
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
        
        # === FISHER EXTREME EXIT ===
        if in_position and position_side > 0 and fisher_extreme_short:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher_extreme_long:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_1w_bearish and price_below_sma200:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_1w_bullish and price_above_sma200:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_1w_bullish or hma_1w_neutral):
                desired_signal = position_size
            elif position_side < 0 and (hma_1w_bearish or hma_1w_neutral):
                desired_signal = -position_size
        
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
        prev_fisher_signal = fisher_signal[i]
    
    return signals