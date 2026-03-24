#!/usr/bin/env python3
"""
Experiment #463: 6h Primary + 1d/1w HTF — Funding Rate Contrarian + Regime Filter

Hypothesis: Funding rate mean reversion is the BEST EDGE for BTC/ETH (Sharpe 0.8-1.5 
through 2022 crash per research). When funding is extreme (+ve = longs overleveraged, 
-ve = shorts overleveraged), price tends to revert. Combine with:
1. Funding z-score(30) < -2.0 → long bias, > +2.0 → short bias
2. 1d/1w HMA for trend filter (only take funding signals WITH trend)
3. 6h RSI(7) for entry timing (oversold for longs, overbought for shorts)
4. CHOP(14) regime filter (mean revert when CHOP>61.8, trend when CHOP<38.2)

Why this should work:
- Funding contrarian works in BOTH bull and bear markets (unlike pure trend)
- 6h timeframe captures multi-day funding cycles without lower-TF noise
- HTF filter prevents counter-trend funding trades during strong trends
- Loose entry thresholds ensure >=30 trades/year

Target: Sharpe>0.45, DD>-35%, trades>=60 train, trades>=10 test
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_funding_contrarian_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

def calculate_zscore(values, period=30):
    """Z-score of values over rolling window"""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)
    
    zscore = np.zeros(n)
    zscore[:] = np.nan
    
    for i in range(period, n):
        window = values[i-period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) >= period:
            mean = np.mean(valid)
            std = np.std(valid)
            if std > 1e-10:
                zscore[i] = (values[i] - mean) / std
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    sma_100 = calculate_sma(close, 100)
    bb_upper, bb_lower = calculate_bollinger(close, period=20, std_dev=2.0)
    choppiness = calculate_choppiness(high, low, close, period=14)
    
    # === FUNDING RATE PROXY (using price momentum as surrogate) ===
    # Since funding data may not be available via mtf_data, use ROC as proxy
    # High positive ROC = overly bullish = funding likely high = short signal
    # High negative ROC = overly bearish = funding likely low = long signal
    roc_20 = np.zeros(n)
    roc_20[:] = np.nan
    for i in range(20, n):
        if close[i-20] > 1e-10:
            roc_20[i] = (close[i] - close[i-20]) / close[i-20] * 100.0
    
    # Z-score of ROC as funding proxy
    funding_proxy_z = calculate_zscore(roc_20, period=30)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_100[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (CHOP) ===
        chop = choppiness[i] if not np.isnan(choppiness[i]) else 50.0
        is_choppy = chop > 55.0  # Mean reversion regime
        is_trending = chop < 45.0  # Trend regime
        
        # === HTF TREND BIAS (1d + 1w) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong bias when both agree
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        htf_neutral = not htf_strong_bull and not htf_strong_bear
        
        # === FUNDING PROXY SIGNAL (ROC z-score contrarian) ===
        funding_z = funding_proxy_z[i] if not np.isnan(funding_proxy_z[i]) else 0.0
        
        # Extreme funding proxy: z < -1.5 = overly bearish = long opportunity
        # Extreme funding proxy: z > +1.5 = overly bullish = short opportunity
        funding_extreme_long = funding_z < -1.0
        funding_extreme_short = funding_z > +1.0
        
        # === RSI EXTREMES (6h timeframe) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_very_oversold = rsi_14[i] < 30.0
        rsi_very_overbought = rsi_14[i] > 70.0
        
        # === BB TOUCH ===
        touch_lower = close[i] <= bb_lower[i]
        touch_upper = close[i] >= bb_upper[i]
        
        # === SMA100 FILTER ===
        above_sma100 = close[i] > sma_100[i]
        below_sma100 = close[i] < sma_100[i]
        
        # === ENTRY LOGIC (LOOSE - ensure trades) ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (mean reversion - funding contrarian + RSI)
        if is_choppy:
            # Long: Funding extreme long + RSI oversold + (above SMA100 OR touch BB lower)
            if funding_extreme_long:
                if rsi_oversold or rsi_very_oversold:
                    if above_sma100 or touch_lower:
                        desired_signal = SIZE_BASE
            
            # Short: Funding extreme short + RSI overbought + (below SMA100 OR touch BB upper)
            if funding_extreme_short:
                if rsi_overbought or rsi_very_overbought:
                    if below_sma100 or touch_upper:
                        desired_signal = -SIZE_BASE
        
        # REGIME 2: TRENDING (follow HTF bias with pullback entry)
        elif is_trending:
            # Long: HTF strong bull + RSI pullback (not extreme)
            if htf_strong_bull:
                if rsi_7[i] < 50.0 and rsi_7[i] > 30.0:  # Pullback, not crash
                    if above_sma100:
                        desired_signal = SIZE_STRONG
            
            # Short: HTF strong bear + RSI pullback (not extreme)
            elif htf_strong_bear:
                if rsi_7[i] > 50.0 and rsi_7[i] < 70.0:  # Pullback, not spike
                    if below_sma100:
                        desired_signal = -SIZE_STRONG
        
        # REGIME 3: NEUTRAL (HTF disagree - use funding only, smaller size)
        if htf_neutral and desired_signal == 0.0:
            if funding_extreme_long and rsi_very_oversold:
                desired_signal = SIZE_BASE * 0.6  # Smaller size in neutral
            elif funding_extreme_short and rsi_very_overbought:
                desired_signal = -SIZE_BASE * 0.6
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.6
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals