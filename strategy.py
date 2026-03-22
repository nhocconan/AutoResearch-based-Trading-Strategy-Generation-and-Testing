#!/usr/bin/env python3
"""
Experiment #483: 1h Multi-Timeframe Regime-Adaptive with Volume Confirmation

Hypothesis: After analyzing 482 failed experiments, the key insight is that 1h timeframe
needs a balance between responsiveness and noise filtering. This strategy combines:

1. 4H HMA(21) TREND BIAS (via mtf_data helper):
   - Provides smoother trend direction than 1h EMA
   - Less whipsaw than daily HMA for 1h entries
   - Bull: price > 4h HMA, Bear: price < 4h HMA

2. CHOPPINESS INDEX (14) REGIME FILTER on 1h:
   - CHOP > 61.8 = ranging (mean-reversion at BB bounds)
   - CHOP < 38.2 = trending (pullback entries in trend direction)
   - Critical for avoiding trend signals in choppy 1h markets

3. VOLUME CONFIRMATION:
   - Volume > 1.5 * SMA(volume, 20) confirms breakouts
   - Prevents false breakouts on low volume
   - Essential for 1h timeframe noise reduction

4. ASYMMETRIC ENTRY LOGIC:
   - TREND + BULL: RSI(14) pullback to 40-50, volume confirm → long
   - TREND + BEAR: RSI(14) rally to 50-60, volume confirm → short
   - RANGE: RSI(14) < 35 long, > 65 short at BB bounds

5. ATR(14) TRAILING STOP at 2.5x:
   - Tighter than daily (1h has less volatility per bar)
   - Signal → 0 when price moves 2.5*ATR against position

6. POSITION SIZING: 0.25 discrete
   - Conservative for 1h noise
   - Discrete levels (0.0, ±0.25) minimize fee churn

Why this should work on 1h:
- 4h HMA provides stable trend bias without 1h noise
- Choppiness Index prevents trend chasing in ranges
- Volume filter reduces false 1h breakouts
- Looser RSI thresholds (35/65 vs 30/70) ensure sufficient trades
- Should generate 50-100 trades/year per symbol (enough for Sharpe)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_4h_hma_chop_rsi_vol_bb_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = tr[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop = calculate_choppiness(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_regime = close[i] > hma_4h_aligned[i]
        bear_regime = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS INDEX REGIME ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        
        # === VOLUME CONFIRMATION ===
        volume_confirm = volume[i] > 1.5 * vol_sma[i] if vol_sma[i] > 0 else False
        
        # === REGIME-ADAPTIVE ENTRY LOGIC ===
        new_signal = 0.0
        
        # TRENDING MARKET: Follow 4h trend with pullback entries
        if trending_market:
            if bull_regime:
                # Bull trend: RSI pullback to 40-50 zone
                if 40 <= rsi[i] <= 55:
                    new_signal = SIZE
            elif bear_regime:
                # Bear trend: RSI rally to 50-60 zone
                if 45 <= rsi[i] <= 60:
                    new_signal = -SIZE
        
        # RANGING MARKET: Mean-revert at Bollinger bounds
        elif ranging_market:
            # Long at lower BB with oversold RSI
            if close[i] <= bb_lower[i] and rsi[i] < 40:
                new_signal = SIZE
            # Short at upper BB with overbought RSI
            elif close[i] >= bb_upper[i] and rsi[i] > 60:
                new_signal = -SIZE
        
        # NEUTRAL CHOP (38.2-61.8): Use stricter RSI extremes
        else:
            if rsi[i] < 35:
                new_signal = SIZE
            elif rsi[i] > 65:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals