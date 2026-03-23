#!/usr/bin/env python3
"""
Experiment #383: 1d Primary + 1w HTF — Fisher Transform + Choppiness Regime + HMA

Hypothesis: Daily timeframe with weekly bias reduces noise and fee drag. Previous 4h strategies
over-traded. This uses proven patterns from research:

1. Ehlers Fisher Transform (period=9) - catches reversals better than RSI in bear markets
2. Choppiness Index regime switch - CHOP>61.8=range(mean revert), CHOP<38.2=trend(follow)
3. HMA(21) for trend - faster than EMA, smoother than SMA, proven on daily
4. 1w HTF HMA for primary bias - simpler than dual HTF which over-filtered
5. Asymmetric sizing - reduce size in bear regime (BTC 2025 is -25%)
6. Relaxed Fisher thresholds (-1.5/+1.5 not -2/+2) to ensure trade generation

Target: 20-40 trades/year on 1d, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL individually).
Must beat current best: mtf_4h_triple_regime_crsi_donchian_1d1w_v1 (Sharpe=0.612)

Key innovation: Fisher Transform excels in bear/range markets (2022 crash, 2025 bear)
where simple trend following fails. Choppiness filter prevents trend trades in ranges.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_hma_1w_regime_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster than EMA, smoother than SMA. Proven on daily timeframe.
    """
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
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
    Converts price into Gaussian-like distribution for reversal detection.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Typical price
    typical = (high_s + low_s + 2 * close) / 4.0
    
    # Normalize to -1 to +1 range
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = (typical - lowest) / (highest - lowest + 1e-10)
    normalized = normalized.clip(0.001, 0.999)
    
    # Fisher transform
    fisher_raw = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    
    # Signal line (1-period lag of Fisher)
    fisher = fisher_raw.fillna(0).values
    signal = np.roll(fisher, 1)
    signal[0] = fisher[0]
    
    return fisher, signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # ATR calculation
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - \
            low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (hh_ll + 1e-10)) / np.log10(period)
    
    chop = chop.fillna(50.0).clip(0, 100).values
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness_index(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate and align HTF HMA for bias (1w)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 1d (target 20-40 trades/year)
    REDUCED_SIZE = 0.20  # Smaller size in uncertain regimes
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if np.isnan(chop[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === HTF BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d HMA) ===
        price_above_hma_21 = close[i] > hma_21[i]
        price_below_hma_21 = close[i] < hma_21[i]
        
        # HMA alignment (bullish: 21>50, bearish: 21<50)
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Relaxed from 61.8 to ensure trades
        is_trending = chop[i] < 45.0  # Relaxed from 38.2
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = (fisher[i] > -1.5) and (prev_fisher <= -1.5)
        fisher_cross_down = (fisher[i] < 1.5) and (prev_fisher >= 1.5)
        
        # Fisher extreme reversals
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        use_reduced_size = is_choppy  # Reduce size in choppy markets
        
        current_size = REDUCED_SIZE if use_reduced_size else BASE_SIZE
        
        # LONG SETUP
        long_bias = price_above_hma_1w or hma_bullish
        
        if long_bias:
            if is_trending:
                # Trend following: Fisher cross up + price above HMA
                if fisher_cross_up and price_above_hma_21:
                    desired_signal = current_size
            else:
                # Mean reversion in range: Fisher oversold + RSI low
                if fisher_oversold and rsi_14[i] < 40:
                    desired_signal = current_size
                elif fisher_cross_up and rsi_14[i] < 50:
                    desired_signal = current_size
        
        # SHORT SETUP
        short_bias = price_below_hma_1w or hma_bearish
        
        if short_bias:
            if is_trending:
                # Trend following: Fisher cross down + price below HMA
                if fisher_cross_down and price_below_hma_21:
                    desired_signal = -current_size
            else:
                # Mean reversion in range: Fisher overbought + RSI high
                if fisher_overbought and rsi_14[i] > 60:
                    desired_signal = -current_size
                elif fisher_cross_down and rsi_14[i] > 50:
                    desired_signal = -current_size
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === FISHER EXIT (reversal signal) ===
        if in_position and position_side > 0 and fisher_cross_down:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher_cross_up:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and rsi_14[i] > 70:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30:
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_hma_1w and hma_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w and hma_bullish:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (price_above_hma_1w or hma_bullish):
                desired_signal = current_size
            elif position_side < 0 and (price_below_hma_1w or hma_bearish):
                desired_signal = -current_size
        
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
                # Position flip
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
        
        # Update Fisher tracking for next iteration
        prev_fisher = fisher[i]
        prev_fisher_signal = fisher_signal[i]
        
        signals[i] = desired_signal
    
    return signals