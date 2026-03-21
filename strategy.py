#!/usr/bin/env python3
"""
Experiment #411: 1h Supertrend + 4h HMA Trend + RSI Pullback + Volume Confirmation
Hypothesis: After 400+ failed experiments, the winning pattern is clear:
- HTF trend filter (4h HMA) prevents counter-trend trades that destroy Sharpe
- Supertrend on primary TF (1h) captures trend momentum with ATR-based stops
- RSI pullback entries (not extremes) work better in trending markets
- Volume confirmation filters out fake breakouts
This combines the best elements from mtf_12h_supertrend_daily_hma_rsi_pullback_v2 (Sharpe=0.499)
but adapted for 1h timeframe with 4h HTF reference. Multiple entry paths ensure >=10 trades.
Timeframe: 1h (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Position size: 0.25 discrete (conservative), stoploss 2.0*ATR.
Target: Beat Sharpe=0.499 with >=10 trades/symbol, all symbols Sharpe>0.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_supertrend_4h_hma_rsi_pullback_volume_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator.
    Returns: supertrend_line, supertrend_direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)
    direction[:] = np.nan
    
    # Initialize
    supertrend[period] = upper_band[period]
    direction[period] = 1  # Start bullish
    
    for i in range(period + 1, n):
        if np.isnan(atr[i]):
            supertrend[i] = np.nan
            direction[i] = np.nan
            continue
        
        # If previous trend was bullish
        if direction[i-1] == 1:
            if close[i] > supertrend[i-1]:
                # Stay bullish, update lower band
                supertrend[i] = max(supertrend[i-1], lower_band[i])
                direction[i] = 1
            else:
                # Flip to bearish
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previous trend was bearish
            if close[i] < supertrend[i-1]:
                # Stay bearish, update upper band
                supertrend[i] = min(supertrend[i-1], upper_band[i])
                direction[i] = -1
            else:
                # Flip to bullish
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_sma(volume, period=20):
    """Calculate volume moving average for volume confirmation."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend_line, supertrend_dir = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    volume_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(supertrend_dir[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(volume_sma[i]):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend bias (HTF direction)
        hma_bullish = close[i] > hma_4h_aligned[i]
        hma_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h Supertrend direction
        st_bullish = supertrend_dir[i] == 1
        st_bearish = supertrend_dir[i] == -1
        
        # Volume confirmation (above average = real move)
        volume_confirmed = volume[i] > 1.0 * volume_sma[i]
        
        # RSI pullback levels (not extremes - work better in trends)
        rsi_pullback_long = 35 < rsi[i] < 55  # Pullback in uptrend
        rsi_pullback_short = 45 < rsi[i] < 65  # Pullback in downtrend
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        # RSI extreme for mean reversion (backup entry)
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: 4h HMA bullish + 1h Supertrend bullish + RSI pullback (primary trend follow)
        if hma_bullish and st_bullish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Path 2: 4h HMA bullish + Supertrend bullish + RSI momentum + volume
        elif hma_bullish and st_bullish and rsi_momentum_long and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Path 3: Supertrend bullish + RSI oversold (mean reversion in uptrend)
        elif st_bullish and rsi_oversold and hma_bullish:
            new_signal = SIZE_ENTRY
        # Path 4: 4h HMA bullish + RSI > 50 + volume (simple momentum)
        elif hma_bullish and rsi[i] > 50 and volume_confirmed and rsi[i] < 75:
            new_signal = SIZE_ENTRY
        # Path 5: Supertrend flip bullish + volume spike (breakout entry)
        elif i > 100 and supertrend_dir[i] == 1 and supertrend_dir[i-1] == -1 and volume[i] > 1.5 * volume_sma[i]:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: 4h HMA bearish + 1h Supertrend bearish + RSI pullback (primary trend follow)
        if hma_bearish and st_bearish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h HMA bearish + Supertrend bearish + RSI momentum + volume
        elif hma_bearish and st_bearish and rsi_momentum_short and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Path 3: Supertrend bearish + RSI overbought (mean reversion in downtrend)
        elif st_bearish and rsi_overbought and hma_bearish:
            new_signal = -SIZE_ENTRY
        # Path 4: 4h HMA bearish + RSI < 50 + volume (simple momentum)
        elif hma_bearish and rsi[i] < 50 and volume_confirmed and rsi[i] > 25:
            new_signal = -SIZE_ENTRY
        # Path 5: Supertrend flip bearish + volume spike (breakout entry)
        elif i > 100 and supertrend_dir[i] == -1 and supertrend_dir[i-1] == 1 and volume[i] > 1.5 * volume_sma[i]:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals