#!/usr/bin/env python3
"""
Experiment #033: 1h Fisher Transform + 4h HMA Trend + Volume Confirmation
Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets (research-backed).
Combined with 4h HMA trend filter to avoid counter-trend trades. Volume spike confirmation filters false signals.
ATR trailing stop at 2.5*ATR limits drawdown. Volume-weighted position sizing adjusts to market conditions.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
Position sizing: 0.25 base, 0.35 on high volume, discrete levels to minimize fee churn.
Key innovation: Fisher Transform crosses are more reliable than RSI in bear markets, especially with volume confirmation.
This addresses the 2022 crash problem - Fisher catches reversals early while HMA keeps us aligned with major trend.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_vol_confirm_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - converts price to Gaussian distribution for clearer signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    Particularly effective in bear/range markets for catching reversals.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    # Calculate median price
    median = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        if highest_high == lowest_low:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * (median[i] - lowest_low) / (highest_high - lowest_low) - 1.0
        
        # Clamp to avoid extreme values
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Apply Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with previous value
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher[i] = 0.7 * fisher[i] + 0.3 * fisher_prev[i-1]
        
        fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

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

def calculate_volume_spike(volume, period=20):
    """Detect volume spikes relative to recent average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_std = vol_s.rolling(window=period, min_periods=period).std().values
    vol_zscore = (volume - vol_avg) / (vol_std + 1e-10)
    vol_zscore[np.isnan(vol_zscore)] = 0.0
    return vol_zscore, vol_avg

def calculate_rsi(close, period=14):
    """Calculate RSI for additional confirmation."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_ema(close, period):
    """Calculate EMA."""
    close_s = pd.Series(close)
    return close_s.ewm(span=period, min_periods=period, adjust=False).mean().values

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
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, 14)
    vol_zscore, vol_avg = calculate_volume_spike(volume, 20)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HIGH = 0.35
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Fisher crossing thresholds
    FISHER_LONG_THRESHOLD = -1.5
    FISHER_SHORT_THRESHOLD = 1.5
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - primary filter
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # Fisher Transform signals
        fisher_cross_up = fisher_prev[i] < FISHER_LONG_THRESHOLD and fisher[i] >= FISHER_LONG_THRESHOLD
        fisher_cross_down = fisher_prev[i] > FISHER_SHORT_THRESHOLD and fisher[i] <= FISHER_SHORT_THRESHOLD
        
        # Volume confirmation (spike = real move)
        vol_confirmed = vol_zscore[i] > 0.5  # Above average volume
        
        # RSI confirmation (avoid extreme overbought/oversold counter-trend)
        rsi_not_extreme_long = rsi[i] < 70  # Not overbought for long
        rsi_not_extreme_short = rsi[i] > 30  # Not oversold for short
        
        # EMA trend confirmation on 1h
        ema_bullish = close[i] > ema_21[i] and close[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and close[i] < ema_50[i]
        
        # Price above/below 200 EMA for major trend
        above_200 = close[i] > ema_200[i]
        below_200 = close[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Fisher cross up + 4h bull trend + volume confirmed
        if fisher_cross_up and bull_trend and vol_confirmed:
            new_signal = SIZE_HIGH if vol_zscore[i] > 1.0 else SIZE_BASE
        # Secondary: Fisher cross up + 4h bull trend + RSI confirmation + above 200 EMA
        elif fisher_cross_up and bull_trend and rsi_not_extreme_long and above_200:
            new_signal = SIZE_BASE
        # Tertiary: Fisher cross up + EMA bullish + volume confirmed (trend continuation)
        elif fisher_cross_up and ema_bullish and vol_confirmed:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: Fisher cross down + 4h bear trend + volume confirmed
        if fisher_cross_down and bear_trend and vol_confirmed:
            new_signal = -SIZE_HIGH if vol_zscore[i] > 1.0 else -SIZE_BASE
        # Secondary: Fisher cross down + 4h bear trend + RSI confirmation + below 200 EMA
        elif fisher_cross_down and bear_trend and rsi_not_extreme_short and below_200:
            new_signal = -SIZE_BASE
        # Tertiary: Fisher cross down + EMA bearish + volume confirmed (trend continuation)
        elif fisher_cross_down and ema_bearish and vol_confirmed:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for more room)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for more room)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals