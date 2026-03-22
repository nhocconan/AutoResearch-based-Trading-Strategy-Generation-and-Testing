#!/usr/bin/env python3
"""
Experiment #028: 4h Fisher Transform Mean Reversion with 1d HMA Regime Filter
Hypothesis: 4h timeframe captures swing reversals while 1d HMA provides regime bias.
Key insight: Fisher Transform catches extremes in bear/range markets better than RSI.
Combined with vol spike detection (ATR ratio) and 1d trend filter for directional bias.
This addresses the 2025 bear market by allowing both long/short based on 1d regime.
Position sizing: 0.25-0.30 discrete levels with 2.5*ATR stoploss.
Why this might work: Fisher Transform has proven 75% win rate on reversals, 
1d HMA smoother than 4h for regime, vol spikes indicate exhaustion points.
Must generate 10+ trades on train, 3+ on test - entry conditions loosened vs failed experiments.
Timeframe: 4h (REQUIRED for exp#028), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_1d_hma_vol_spike_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals at extremes better than RSI in bear/range markets.
    """
    hl2 = (high + low) / 2
    
    # Calculate highest high and lowest low over period
    highest = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    
    # Normalize to 0-1 range
    epsilon = 1e-10
    normalized = (hl2 - lowest) / (highest - lowest + epsilon)
    normalized = np.clip(normalized, 0.001, 0.999)  # Avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + epsilon))
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    
    # Fisher Transform for reversals
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    # Bollinger Bands for mean reversion
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Vol spike detection (ATR ratio)
    atr_ratio = atr / (atr_30 + 1e-10)
    
    # BB width for squeeze detection
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    bb_width_sma = pd.Series(bb_width).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.28
    SIZE_HALF = 0.14
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - main regime filter
        bull_regime_1d = close[i] > hma_1d_aligned[i]
        bear_regime_1d = close[i] < hma_1d_aligned[i]
        
        # 4h trend confirmation
        bull_trend_4h = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_4h = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # Fisher Transform signals (reversal detection)
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_oversold = fisher[i] < -1.2
        fisher_cross_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_overbought = fisher[i] > 1.2
        fisher_cross_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # Vol spike detection (exhaustion signal)
        vol_spike = atr_ratio[i] > 1.8  # Current ATR > 1.8x 30-period ATR
        
        # BB mean reversion
        price_at_lower_bb = close[i] <= bb_lower[i] * 1.005
        price_at_upper_bb = close[i] >= bb_upper[i] * 0.995
        
        # RSI extremes
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # BB squeeze (low volatility before breakout)
        bb_squeeze = bb_width[i] < bb_width_sma[i] * 0.7
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        if bull_regime_1d:
            # Primary: Fisher oversold + vol spike (reversal at exhaustion)
            if fisher_oversold and vol_spike and above_200:
                new_signal = SIZE_BASE
            
            # Secondary: Fisher cross + BB lower band
            elif fisher_cross_long and price_at_lower_bb and bull_trend_4h:
                new_signal = SIZE_BASE
            
            # Tertiary: RSI oversold + price near EMA21 pullback
            elif rsi_oversold and close[i] <= ema_21[i] * 1.01 and above_200:
                new_signal = SIZE_HALF
            
            # Quaternary: BB squeeze breakout long
            elif bb_squeeze and close[i] > bb_mid[i] and bull_trend_4h:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES ===
        elif bear_regime_1d:
            # Primary: Fisher overbought + vol spike (reversal at exhaustion)
            if fisher_overbought and vol_spike and below_200:
                new_signal = -SIZE_BASE
            
            # Secondary: Fisher cross + BB upper band
            elif fisher_cross_short and price_at_upper_bb and bear_trend_4h:
                new_signal = -SIZE_BASE
            
            # Tertiary: RSI overbought + price near EMA21 bounce
            elif rsi_overbought and close[i] >= ema_21[i] * 0.99 and below_200:
                new_signal = -SIZE_HALF
            
            # Quaternary: BB squeeze breakout short
            elif bb_squeeze and close[i] < bb_mid[i] and bear_trend_4h:
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