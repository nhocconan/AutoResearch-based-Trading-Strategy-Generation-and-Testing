#!/usr/bin/env python3
"""
Experiment #565: 15m Mean Reversion with 4h HMA Trend Filter

Hypothesis: After 500+ failed experiments, the key insight for 15m timeframe:
1. 15m is TOO NOISY for pure trend following (exp #553, #559 both failed badly)
2. Mean reversion works better on intraday timeframes than trend following
3. 4h HMA provides proven trend bias filter (used in current best strategy)
4. RSI extremes (loose thresholds <35/>65) ensure we get trades (avoid 0-trade failure)
5. Volume confirmation filters false breakouts without being too restrictive
6. Simple logic = more trades = statistical significance

Why this should work on 15m:
- 15m has 96 bars/day = high frequency but manageable with HTF filter
- RSI mean reversion captures intraday oversold/overbought conditions
- 4h HMA trend bias prevents counter-trend entries (major failure mode)
- Loose RSI thresholds (35/65 not 25/75) ensure we generate trades
- 2.0*ATR stoploss protects against adverse moves
- Volume filter avoids low-liquidity fakeouts

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (max 0.40)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_meanrev_4h_hma_volume_atr_v1"
timeframe = "15m"
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
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average for volume confirmation."""
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=period, min_periods=period).mean()
    return vol_ma.values

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
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ma_20 = calculate_volume_ma(volume, 20)
    
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
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === 15M RSI MEAN REVERSION (loose thresholds for trades) ===
        oversold = rsi_14[i] < 35  # Loose threshold to ensure trades
        overbought = rsi_14[i] > 65  # Loose threshold to ensure trades
        
        # === VOLUME CONFIRMATION ===
        volume_confirm = volume[i] > vol_ma_20[i] * 0.8  # At least 80% of avg volume
        
        # === ENTRY LOGIC (SIMPLE to ensure trades) ===
        new_signal = 0.0
        
        # Long: RSI oversold + 4h bullish bias + volume confirm
        if oversold and bull_bias and volume_confirm:
            new_signal = SIZE
        
        # Short: RSI overbought + 4h bearish bias + volume confirm
        elif overbought and bear_bias and volume_confirm:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
                new_signal = 0.0
        
        # === RSI EXIT (mean reversion complete) ===
        # Exit long when RSI recovers to neutral
        if in_position and position_side > 0 and rsi_14[i] > 55:
            new_signal = 0.0
        
        # Exit short when RSI drops to neutral
        if in_position and position_side < 0 and rsi_14[i] < 45:
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