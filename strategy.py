#!/usr/bin/env python3
"""
Experiment #007: 15m Fisher Transform Reversal with 1h/4h HMA Trend Filter
Hypothesis: 15m timeframe captures intraday reversals using Fisher Transform (Ehlers),
which converts price to near-Gaussian distribution for clearer turning point detection.
HTF (1h + 4h) HMA provides trend bias to avoid counter-trend entries.
Volume confirmation filters false breakouts. ATR trailing stop limits drawdown.
Key innovation: Fisher Transform excels in range/bear markets (2022 crash, 2025 range)
where simple EMA/RSI strategies failed. Dual HTF filter (1h + 4h) reduces whipsaws.
Position sizing: 0.20 base, 0.30 max, discrete levels to minimize fee churn.
Stoploss: 2.5*ATR trailing stop. Timeframe: 15m (REQUIRED), HTF: 1h + 4h via mtf_data.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_1h4h_hma_vol_v1"
timeframe = "15m"
leverage = 1.0

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - converts price to Gaussian distribution.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Calculate median price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize price
        if highest != lowest:
            normalized = 0.66 * ((hl2 - lowest) / (highest - lowest) - 0.5) + 0.67 * 0.0
            if i > period:
                normalized = 0.66 * ((hl2 - lowest) / (highest - lowest) - 0.5) + 0.67 * prev_normalized
            else:
                normalized = 0.0
            
            # Clamp to avoid division errors
            normalized = np.clip(normalized, -0.99, 0.99)
            
            # Fisher calculation
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
            if i > period:
                trigger[i] = fisher[i-1]
            else:
                trigger[i] = fisher[i]
            
            prev_normalized = normalized
    
    return fisher, trigger

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

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above threshold * average volume."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    volume_ratio = volume / vol_sma
    volume_ratio[np.isnan(volume_ratio)] = 0.0
    return volume_ratio > threshold

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    fisher, trigger = calculate_fisher_transform(high, low, 9)
    volume_spike = calculate_volume_spike(volume, 20, 1.5)
    
    # Additional trend filters
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_MAX = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(trigger[i]):
            signals[i] = 0.0
            continue
        
        # HTF trend bias (1h + 4h) - both must agree for strong signal
        bull_trend_1h = close[i] > hma_1h_aligned[i]
        bear_trend_1h = close[i] < hma_1h_aligned[i]
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # Strong trend: both 1h and 4h agree
        strong_bull = bull_trend_1h and bull_trend_4h
        strong_bear = bear_trend_1h and bear_trend_4h
        
        # Moderate trend: at least one agrees
        mod_bull = bull_trend_1h or bull_trend_4h
        mod_bear = bear_trend_1h or bear_trend_4h
        
        # Fisher Transform signals
        fisher_cross_long = fisher[i] > -1.5 and trigger[i] <= -1.5 if i > 0 else False
        fisher_cross_short = fisher[i] < 1.5 and trigger[i] >= 1.5 if i > 0 else False
        
        # Fisher extreme levels (reversal zones)
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # RSI confirmation
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_rising = rsi[i] > rsi[i-3] if i >= 3 else False
        rsi_falling = rsi[i] < rsi[i-3] if i >= 3 else False
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_21[i] and close[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and close[i] < ema_50[i]
        ema_bullish_strong = ema_bullish and close[i] > ema_200[i]
        ema_bearish_strong = ema_bearish and close[i] < ema_200[i]
        
        # Volume confirmation
        vol_confirmed = volume_spike[i]
        
        new_signal = 0.0
        
        # === STRONG LONG: Fisher cross + strong HTF bull + volume + RSI ===
        if fisher_cross_long and strong_bull and ema_bullish and rsi_rising:
            new_signal = SIZE_MAX if vol_confirmed else SIZE_BASE
        
        # === STRONG SHORT: Fisher cross + strong HTF bear + volume + RSI ===
        elif fisher_cross_short and strong_bear and ema_bearish and rsi_falling:
            new_signal = -SIZE_MAX if vol_confirmed else -SIZE_BASE
        
        # === MODERATE LONG: Fisher oversold + moderate HTF bull ===
        elif fisher_oversold and mod_bull and rsi_oversold:
            new_signal = SIZE_BASE
        
        # === MODERATE SHORT: Fisher overbought + moderate HTF bear ===
        elif fisher_overbought and mod_bear and rsi_overbought:
            new_signal = -SIZE_BASE
        
        # === CONSERVATIVE: EMA + Fisher alignment ===
        elif fisher_cross_long and ema_bullish_strong:
            new_signal = SIZE_HALF
        
        elif fisher_cross_short and ema_bearish_strong:
            new_signal = -SIZE_HALF
        
        # === RSI EXTREME REVERSAL with HTF support ===
        elif rsi_oversold and bull_trend_4h and fisher[i] < -1.0:
            new_signal = SIZE_HALF
        
        elif rsi_overbought and bear_trend_4h and fisher[i] > 1.0:
            new_signal = -SIZE_HALF
        
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