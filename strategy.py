#!/usr/bin/env python3
"""
Experiment #046: 4h Volatility Spike Mean Reversion with 1d HMA Regime Filter
Hypothesis: Volatility spikes (ATR ratio > 2.0) combined with BB extremes capture panic reversals.
Key insight: BTC/ETH crash hard then recover quickly. Vol spike + oversold = long opportunity in bull regime.
1d HMA provides regime filter to avoid counter-trend trades. 4h timeframe balances signal frequency vs noise.
Position sizing: 0.25 discrete levels, stoploss at 2.5*ATR, take profit at 2R.
Timeframe: 4h (REQUIRED for exp#046), HTF: 1d via mtf_data helper.
Why this might work: Vol mean reversion works in all regimes, 1d filter prevents fighting major trend.
Must generate 10+ trades on train, 3+ on test - conditions loosened vs failed vol breakout strategies.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_spike_1d_hma_meanrev_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with wider std for extreme detection."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_stoch_rsi(close, period=14):
    """Calculate Stochastic RSI for faster mean reversion signals."""
    rsi = calculate_rsi(close, period)
    rsi_min = pd.Series(rsi).rolling(window=period, min_periods=period).min().values
    rsi_max = pd.Series(rsi).rolling(window=period, min_periods=period).max().values
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
    return stoch_rsi

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
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    stoch_rsi = calculate_stoch_rsi(close, 14)
    
    # Bollinger Bands with wider bands for extreme detection
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    
    # Z-score for mean reversion
    zscore_20 = calculate_zscore(close, 20)
    
    # EMA/SMA for trend
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Volatility ratio (ATR spike detector)
    vol_ratio = atr_7 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    take_profit_level = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - main regime filter
        bull_regime_1d = close[i] > hma_1d_aligned[i]
        bear_regime_1d = close[i] < hma_1d_aligned[i]
        
        # Volatility spike detection
        vol_spike = vol_ratio[i] > 1.8  # ATR(7) > 1.8x ATR(30)
        vol_extreme = vol_ratio[i] > 2.5  # Extreme spike
        
        # Price at BB extreme
        at_bb_lower = close[i] <= bb_lower[i] * 1.005  # At or below lower band
        at_bb_upper = close[i] >= bb_upper[i] * 0.995  # At or above upper band
        
        # RSI extremes (loosened for more trades)
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_oversold = rsi_7[i] < 25
        rsi_extreme_overbought = rsi_7[i] > 75
        
        # Stoch RSI extremes
        stoch_oversold = stoch_rsi[i] < 0.15
        stoch_overbought = stoch_rsi[i] > 0.85
        
        # Z-score extremes
        zscore_oversold = zscore_20[i] < -1.8
        zscore_overbought = zscore_20[i] > 1.8
        
        # Trend confirmation on 4h
        above_ema21 = close[i] > ema_21[i]
        below_ema21 = close[i] < ema_21[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (vol spike mean reversion in bull regime) ===
        if bull_regime_1d:
            # Primary: Vol spike + BB lower + RSI oversold
            if vol_spike and at_bb_lower and rsi_oversold:
                new_signal = SIZE_BASE
            
            # Secondary: Extreme vol + extreme RSI (panic bottom)
            elif vol_extreme and rsi_extreme_oversold and zscore_oversold:
                new_signal = SIZE_BASE
            
            # Tertiary: Stoch RSI oversold + vol spike
            elif vol_spike and stoch_oversold and above_sma200:
                new_signal = SIZE_HALF
            
            # Quaternary: BB lower + RSI oversold (no vol spike needed)
            elif at_bb_lower and rsi_14[i] < 40 and bull_regime_1d:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (vol spike mean reversion in bear regime) ===
        elif bear_regime_1d:
            # Primary: Vol spike + BB upper + RSI overbought
            if vol_spike and at_bb_upper and rsi_overbought:
                new_signal = -SIZE_BASE
            
            # Secondary: Extreme vol + extreme RSI (panic top)
            elif vol_extreme and rsi_extreme_overbought and zscore_overbought:
                new_signal = -SIZE_BASE
            
            # Tertiary: Stoch RSI overbought + vol spike
            elif vol_spike and stoch_overbought and below_sma200:
                new_signal = -SIZE_HALF
            
            # Quaternary: BB upper + RSI overbought (no vol spike needed)
            elif at_bb_upper and rsi_14[i] > 60 and bear_regime_1d:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss and take profit
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Trailing stop at 2.5*ATR
            current_stop = highest_close - 2.5 * atr_14[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Take profit at 2R (reduce to half position)
            profit = close[i] - entry_price
            risk = entry_price - (entry_price - 2.5 * atr_14[int(np.where(atr_14[:i+1] > 0)[0][-1]) if len(np.where(atr_14[:i+1] > 0)[0]) > 0 else i])
            if risk > 0 and profit >= 2.0 * risk and new_signal == 0.0:
                new_signal = SIZE_HALF  # Take partial profit
            
            # Stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss and take profit
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Trailing stop at 2.5*ATR
            current_stop = lowest_close + 2.5 * atr_14[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Take profit at 2R (reduce to half position)
            profit = entry_price - close[i]
            risk = (entry_price + 2.5 * atr_14[int(np.where(atr_14[:i+1] > 0)[0][-1]) if len(np.where(atr_14[:i+1] > 0)[0]) > 0 else i]) - entry_price
            if risk > 0 and profit >= 2.0 * risk and new_signal == 0.0:
                new_signal = -SIZE_HALF  # Take partial profit
            
            # Stoploss hit
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