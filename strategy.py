#!/usr/bin/env python3
"""
Experiment #068: 30m Choppiness Index Regime + 4h HMA Trend Filter + Volume Confirmation
Hypothesis: 30m timeframe needs strong regime detection to avoid whipsaw. Choppiness Index (CHOP)
distinguishes trending (CHOP<38.2) vs ranging (CHOP>61.8) markets. In trending regimes, follow
4h HMA direction with pullback entries. In ranging regimes, mean-revert at Bollinger bands.
Volume confirmation (1.5x 20-bar avg) filters false breakouts. ATR stoploss at 2.5x.
Why this might work: CHOP is proven regime filter from Australian trader research. Combines
trend-following (trending) with mean-reversion (ranging) - adapts to market conditions.
4h HMA provides HTF trend bias without excessive lag. Volume confirms genuine moves.
Position sizing: 0.25 base, 0.35 strong trend, discrete levels to minimize fee churn.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_regime_4h_hma_vol_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection indicator.
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    Source: Australian trader E.W. Dreiss
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (close - lower) / (upper - lower + 1e-10)
    return upper, lower, pct_b

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    return vol_ratio

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = 1
        else:
            if close[i] > supertrend[i - 1]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
    
    return supertrend, trend

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    chop = calculate_choppiness_index(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    bb_upper, bb_lower, pct_b = calculate_bollinger_bands(close, 20, 2.0)
    vol_ratio = calculate_volume_ratio(volume, 20)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_trend[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = intermediate trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # EMA alignment on 30m
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # Supertrend direction
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # === REGIME DETECTION ===
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        neutral_regime = 38.2 <= chop[i] <= 61.8
        
        # Volume confirmation
        high_volume = vol_ratio[i] > 1.5
        normal_volume = 0.8 <= vol_ratio[i] <= 1.5
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 40 <= rsi[i] <= 60
        rsi_extreme_oversold = rsi[i] < 25
        rsi_extreme_overbought = rsi[i] > 75
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = pct_b[i] < 0.1
        near_bb_upper = pct_b[i] > 0.9
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / close[i] < 0.05
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        
        # Path 1: Trending regime + 4h bullish + pullback entry
        if trending_regime and bull_trend_4h:
            if ema_bullish and st_bullish:
                if rsi[i] > 40 and rsi[i] < 65:
                    if close[i] > ema_21[i]:
                        if high_volume or normal_volume:
                            new_signal = SIZE_BASE
                        if rsi[i] < 50 and ema_21[i] > ema_50[i]:
                            new_signal = SIZE_STRONG
        
        # Path 2: Ranging regime + mean reversion at BB lower
        if ranging_regime:
            if near_bb_lower and rsi_oversold:
                if bull_trend_4h:  # Only long if HTF bullish
                    new_signal = SIZE_HALF
            elif rsi_extreme_oversold and close[i] < bb_lower[i]:
                new_signal = SIZE_HALF
        
        # Path 3: Supertrend flip bullish + volume confirmation
        if st_bullish and i > 200:
            if st_trend[i - 1] == -1:  # Just flipped
                if bull_trend_4h and high_volume:
                    new_signal = SIZE_BASE
        
        # Path 4: EMA crossover + HTF confirmation
        if ema_bullish and bull_trend_4h:
            if rsi_neutral and close[i] > ema_21[i]:
                if st_bullish:
                    new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        
        # Path 1: Trending regime + 4h bearish + pullback entry
        if trending_regime and bear_trend_4h:
            if ema_bearish and st_bearish:
                if rsi[i] > 35 and rsi[i] < 60:
                    if close[i] < ema_21[i]:
                        if high_volume or normal_volume:
                            new_signal = -SIZE_BASE
                        if rsi[i] > 50 and ema_21[i] < ema_50[i]:
                            new_signal = -SIZE_STRONG
        
        # Path 2: Ranging regime + mean reversion at BB upper
        if ranging_regime:
            if near_bb_upper and rsi_overbought:
                if bear_trend_4h:  # Only short if HTF bearish
                    new_signal = -SIZE_HALF
            elif rsi_extreme_overbought and close[i] > bb_upper[i]:
                new_signal = -SIZE_HALF
        
        # Path 3: Supertrend flip bearish + volume confirmation
        if st_bearish and i > 200:
            if st_trend[i - 1] == 1:  # Just flipped
                if bear_trend_4h and high_volume:
                    new_signal = -SIZE_BASE
        
        # Path 4: EMA crossover + HTF confirmation
        if ema_bearish and bear_trend_4h:
            if rsi_neutral and close[i] < ema_21[i]:
                if st_bearish:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals