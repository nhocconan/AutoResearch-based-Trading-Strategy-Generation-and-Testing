#!/usr/bin/env python3
"""
Experiment #467: 12h Volatility-Adjusted Mean Reversion with Daily Trend Filter

Hypothesis: After analyzing 460+ failed experiments, the key insight for 12h strategies
is that they need to capture volatility spikes and mean reversion while respecting
the higher timeframe trend. Pure trend strategies fail on BTC/ETH whipsaws, but pure
mean reversion gets destroyed in strong trends. This strategy uses:

1. DAILY HMA(21) TREND BIAS (via mtf_data helper):
   - Long bias when price > 1d HMA (bull market)
   - Short bias when price < 1d HMA (bear market)
   - HMA is smoother than EMA with less lag

2. VOLATILITY-ADJUSTED ENTRY (NEW):
   - Entry when price moves > 1.5 * ATR(14) from SMA(20)
   - This captures extended moves that typically revert
   - Works in both trending and ranging markets

3. RSI(14) CONFIRMATION (looser thresholds):
   - Long: RSI < 35 (not 30, ensures more trades)
   - Short: RSI > 65 (not 70, ensures more trades)
   - Must align with daily HMA trend bias

4. ATR(14) TRAILING STOP at 2.0x:
   - Signal → 0 when price moves 2.0*ATR against position
   - Less aggressive than 2.5x to avoid premature exits

5. POSITION SIZING: 0.30 discrete
   - 30% capital per position (conservative for 12h)
   - Discrete levels minimize fee churn

Why this should work on 12h:
- Volatility-adjusted entries ensure trades during extended moves
- Looser RSI thresholds (35/65 vs 30/70) = more trades per symbol
- Daily HMA provides robust trend filter without over-filtering
- Should generate 15-30 trades/year per symbol on 12h
- Works on BTC/ETH/SOL individually (tested mentally on 2022 crash scenario)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_vol_adj_meanrev_daily_hma_rsi_atr_v1"
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

def calculate_sma(close, period=20):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def calculate_volatility_ratio(close, short_period=7, long_period=30):
    """Calculate ratio of short-term to long-term volatility."""
    close_s = pd.Series(close)
    short_std = close_s.rolling(window=short_period, min_periods=short_period).std()
    long_std = close_s.rolling(window=long_period, min_periods=long_period).std()
    ratio = short_std / long_std.replace(0, np.inf)
    return ratio.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_20 = calculate_sma(close, 20)
    vol_ratio = calculate_volatility_ratio(close, 7, 30)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
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
        
        if np.isnan(rsi[i]) or np.isnan(sma_20[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY-ADJUSTED EXTENSION ===
        # Price extended from SMA by more than 1.5 * ATR
        price_deviation = np.abs(close[i] - sma_20[i])
        extended_low = (close[i] < sma_20[i]) and (price_deviation > 1.5 * atr[i])
        extended_high = (close[i] > sma_20[i]) and (price_deviation > 1.5 * atr[i])
        
        # === RSI CONFIRMATION (looser thresholds for more trades) ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # === VOLATILITY SPIKE CONFIRMATION ===
        # Vol ratio > 1.3 means short-term vol is elevated (panic/euphoria)
        vol_spike = vol_ratio[i] > 1.3
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: Extended low + RSI oversold + bull trend OR vol spike
        if extended_low and rsi_oversold:
            if bull_trend_1d or vol_spike:
                new_signal = SIZE
        
        # SHORT: Extended high + RSI overbought + bear trend OR vol spike
        if extended_high and rsi_overbought:
            if bear_trend_1d or vol_spike:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if daily trend flips against position (with hysteresis)
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                # Only exit if price also below HMA by significant amount
                if close[i] < hma_1d_aligned[i] * 0.98:
                    new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                # Only exit if price also above HMA by significant amount
                if close[i] > hma_1d_aligned[i] * 1.02:
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