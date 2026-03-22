#!/usr/bin/env python3
"""
Experiment #004: 4h Multi-Regime Strategy with 1d HMA Trend Filter
Hypothesis: 4h timeframe balances swing trading with reduced noise vs lower TFs.
1d HMA provides major trend bias. Key innovation: Volatility regime detection via ATR ratio
(ATR7/ATR30) separates panic spikes (mean revert) from normal conditions (trend follow).
RSI(7) extremes capture pullbacks in trend direction. Donchian(20) breakouts with HTF confirmation
for momentum entries. Asymmetric logic: more aggressive longs in bull, shorts only on strong signals in bear.
Position sizing: 0.25 base, 0.35 max, discrete levels. Stoploss: 2.5*ATR trailing.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volregime_1d_hma_rsi_v1"
timeframe = "4h"
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

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    return upper, lower, sma, bandwidth

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    bb_upper, bb_lower, bb_sma, bb_bw = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volatility regime: ATR ratio
    atr_ratio = np.zeros(n)
    atr_ratio[:] = np.nan
    mask = atr_30 > 0
    atr_ratio[mask] = atr_7[mask] / atr_30[mask]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF)
        bull_trend_1d = close[i] > hma_1d_21_aligned[i] and hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        bear_trend_1d = close[i] < hma_1d_21_aligned[i] and hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        neutral_1d = not bull_trend_1d and not bear_trend_1d
        
        # 4h trend
        bull_trend_4h = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_4h = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Volatility regime
        vol_spike = not np.isnan(atr_ratio[i]) and atr_ratio[i] > 2.0
        vol_normal = not np.isnan(atr_ratio[i]) and atr_ratio[i] < 1.3
        vol_compress = not np.isnan(atr_ratio[i]) and atr_ratio[i] < 0.7
        
        # RSI signals (7-period for faster response)
        rsi_oversold = rsi_7[i] < 35
        rsi_overbought = rsi_7[i] > 65
        rsi_extreme_oversold = rsi_7[i] < 25
        rsi_extreme_overbought = rsi_7[i] > 75
        rsi_rising = rsi_7[i] > rsi_7[i - 3] if i >= 3 else False
        rsi_falling = rsi_7[i] < rsi_7[i - 3] if i >= 3 else False
        
        # Donchian breakout (use previous bar to avoid look-ahead)
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # Bollinger signals
        price_below_lower = close[i] < bb_lower[i] * 1.005
        price_above_upper = close[i] > bb_upper[i] * 0.995
        price_near_sma = abs(close[i] - bb_sma[i]) / bb_sma[i] < 0.01
        
        # HMA crossover
        hma_cross_long = hma_21[i] > hma_50[i] and hma_21[i - 1] <= hma_50[i - 1] if i >= 1 else False
        hma_cross_short = hma_21[i] < hma_50[i] and hma_21[i - 1] >= hma_50[i - 1] if i >= 1 else False
        
        # EMA alignment
        ema_aligned_bull = ema_21[i] > ema_50[i] and ema_50[i] > ema_200[i]
        ema_aligned_bear = ema_21[i] < ema_50[i] and ema_50[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === VOL SPIKE REGIME: Mean Reversion (panic selling/buying) ===
        if vol_spike:
            # Long on extreme oversold + price below BB lower
            if rsi_extreme_oversold and price_below_lower and bull_trend_1d:
                new_signal = SIZE_MAX
            elif rsi_extreme_oversold and price_below_lower:
                new_signal = SIZE_BASE
            # Short on extreme overbought + price above BB upper
            elif rsi_extreme_overbought and price_above_upper and bear_trend_1d:
                new_signal = -SIZE_MAX
            elif rsi_extreme_overbought and price_above_upper:
                new_signal = -SIZE_BASE
        
        # === VOL NORMAL/COMPRESS: Trend Following ===
        elif vol_normal or vol_compress:
            # Strong long: breakout + HTF bull + EMA aligned + RSI rising
            if breakout_long and bull_trend_1d and ema_aligned_bull and rsi_rising:
                new_signal = SIZE_MAX
            # Moderate long: breakout + HTF bull
            elif breakout_long and bull_trend_1d:
                new_signal = SIZE_BASE
            # HMA crossover with trend
            elif hma_cross_long and bull_trend_1d and close[i] > ema_200[i]:
                new_signal = SIZE_BASE
            # RSI pullback in uptrend
            elif rsi_oversold and bull_trend_1d and bull_trend_4h and rsi_rising:
                new_signal = SIZE_HALF
            
            # Strong short: breakout + HTF bear + EMA aligned + RSI falling
            if breakout_short and bear_trend_1d and ema_aligned_bear and rsi_falling:
                new_signal = -SIZE_MAX
            # Moderate short: breakout + HTF bear
            elif breakout_short and bear_trend_1d:
                new_signal = -SIZE_BASE
            # HMA crossover with trend
            elif hma_cross_short and bear_trend_1d and close[i] < ema_200[i]:
                new_signal = -SIZE_BASE
            # RSI pullback in downtrend
            elif rsi_overbought and bear_trend_1d and bear_trend_4h and rsi_falling:
                new_signal = -SIZE_HALF
        
        # === NEUTRAL VOL: Conservative entries ===
        else:
            # Only take strongest signals with full confirmation
            if breakout_long and bull_trend_1d and rsi_extreme_oversold:
                new_signal = SIZE_BASE
            elif breakout_short and bear_trend_1d and rsi_extreme_overbought:
                new_signal = -SIZE_BASE
            # BB mean reversion with HTF support
            elif price_below_lower and rsi_oversold and bull_trend_1d:
                new_signal = SIZE_HALF
            elif price_above_upper and rsi_overbought and bear_trend_1d:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr_14[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr_14[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
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