#!/usr/bin/env python3
"""
Experiment #365: 12h Z-Score Mean Reversion with 1d HMA Trend Bias + Volume Filter

Hypothesis: After 354 failed experiments, the pattern shows trend-following breakouts
(Donchian, Supertrend) fail in bear/range markets (2022 crash, 2025 bear). 

Mean reversion with HTF trend filter should work better:
1. Z-SCORE(20): Captures overextended moves - long when z<-1.5, short when z>+1.5
   - More frequent signals than Donchian breakouts
   - Works in both trending AND ranging markets
   
2. 1d HMA TREND BIAS: Only take mean reversion WITH the higher timeframe trend
   - Long z-score extreme only if price > 1d HMA(21) [bullish bias]
   - Short z-score extreme only if price < 1d HMA(21) [bearish bias]
   - Critical: prevents counter-trend mean reversion losses

3. VOLUME CONFIRMATION: Volume must be > 1.2x 20-period average
   - Filters out low-conviction moves
   - Ensures institutional participation

4. ATR TRAILING STOP (2.5x): Protect capital on reversals
   - Signal → 0 when price moves 2.5*ATR against position

5. POSITION SIZING: 0.25 discrete (conservative for 12h)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why 12h + Z-score should work:
- Mean reversion works in bear/range markets (unlike pure trend following)
- 1d HMA filter prevents dangerous counter-trend entries
- Volume confirmation reduces false signals
- Should generate 30-60 trades/year (enough for statistical significance)
- Works on BTC, ETH, AND SOL (not SOL-biased)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_zscore_1d_hma_volume_meanrev_atr_v1"
timeframe = "12h"
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

def calculate_zscore(close, period=20):
    """
    Calculate Z-Score of price relative to rolling mean.
    Z = (price - rolling_mean) / rolling_std
    Z < -1.5 = oversold, Z > +1.5 = overbought
    """
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std
    return zscore.values

def calculate_volume_ratio(volume, period=20):
    """
    Calculate volume ratio relative to 20-period average.
    Ratio > 1.2 = above average volume (institutional interest)
    """
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume / vol_avg.values
    vol_ratio[np.isnan(vol_ratio)] = 1.0
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        # === 1d HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === Z-SCORE MEAN REVERSION SIGNALS ===
        # Loosened from 2.0 to 1.5 to generate more trades
        oversold = zscore[i] < -1.5
        overbought = zscore[i] > 1.5
        
        # === VOLUME CONFIRMATION ===
        # Volume must be above average (institutional interest)
        volume_confirmed = vol_ratio[i] > 1.2
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Z-score oversold + 1d bullish bias + volume confirmed
        if oversold and bull_trend_1d and volume_confirmed:
            new_signal = SIZE
        
        # SHORT ENTRY: Z-score overbought + 1d bearish bias + volume confirmed
        elif overbought and bear_trend_1d and volume_confirmed:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1d trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === Z-SCORE REVERSION EXIT ===
        # Exit long when z-score returns to neutral (>-0.5)
        # Exit short when z-score returns to neutral (<+0.5)
        if in_position and new_signal != 0.0:
            if position_side > 0 and zscore[i] > -0.5:
                new_signal = 0.0
            if position_side < 0 and zscore[i] < 0.5:
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