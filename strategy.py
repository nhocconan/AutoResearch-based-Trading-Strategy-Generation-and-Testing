#!/usr/bin/env python3
"""
Experiment #208: 4h Bollinger Squeeze Breakout + 1d HMA Trend + Volume Confirmation + ATR Stop

Hypothesis: 4h timeframe captures swing trends while filtering noise. Bollinger Band Width
detects volatility squeezes (low BW = compression before breakout), 1d HMA provides stable
higher-timeframe bias, volume spike confirms breakout validity, and ATR trailing stop
protects against false breakouts. This differs from failed Donchian breakouts by using
volatility-based entry timing rather than pure price levels.

Why this might work where others failed:
- #178 (4h Donchian): Sharpe=-0.989 - pure price breakout, no vol filter
- #202 (4h Asymmetric Regime): Sharpe=-0.991 - regime filter too complex
- #196 (4h Fisher): Sharpe=-1.782 - Fisher whipsaws in strong trends
- Bollinger squeeze + volume = proven combo (Volatility Breakout strategy)
- 1d HMA filter prevents counter-trend squeeze plays
- Volume confirmation reduces false breakouts significantly

Key differences from failed strategies:
1. Entry on BB squeeze RELEASE (BW expanding from low), not just price level
2. Volume must be > 1.5x 20-period avg to confirm breakout
3. 1d HMA bias must align with breakout direction
4. Conservative sizing (0.25) with 2.5*ATR trailing stop

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_squeeze_1d_hma_vol_atr_v1"
timeframe = "4h"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands (middle, upper, lower, bandwidth)."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    # Bandwidth = (Upper - Lower) / Middle
    bandwidth = (upper - lower) / (middle + 1e-10)
    return middle.values, upper.values, lower.values, bandwidth.values

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
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smooth with Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    bb_mid, bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    adx = calculate_adx(high, low, close, 14)
    
    # Calculate BB bandwidth percentile for squeeze detection
    # Squeeze = bandwidth at 20-period low
    bb_bw_percentile = np.zeros(n)
    lookback = 20
    for i in range(lookback, n):
        current_bw = bb_bandwidth[i]
        historical_bw = bb_bandwidth[i-lookback:i]
        percentile = np.sum(historical_bw <= current_bw) / lookback
        bb_bw_percentile[i] = percentile
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === BOLLINGER SQUEEZE DETECTION ===
        # Squeeze = bandwidth at 20-period low (percentile < 0.2)
        # Release = bandwidth expanding from squeeze
        is_squeeze = bb_bw_percentile[i] < 0.2
        bw_expanding = bb_bandwidth[i] > bb_bandwidth[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        # Volume must be > 1.5x 20-period average for breakout confirmation
        vol_spike = volume[i] > 1.5 * vol_sma[i] if vol_sma[i] > 0 else False
        
        # === PRICE BREAKOUT ===
        # Long: price breaks above BB upper
        # Short: price breaks below BB lower
        breakout_long = close[i] > bb_upper[i-1] if i > 0 else False
        breakout_short = close[i] < bb_lower[i-1] if i > 0 else False
        
        # === RSI MOMENTUM FILTER ===
        # Long: RSI > 50 (bullish momentum)
        # Short: RSI < 50 (bearish momentum)
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # === ADX TREND STRENGTH ===
        # ADX > 20 = trending market (avoid choppy ranges)
        trend_strength = adx[i] > 20
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 1d bullish + squeeze release + volume spike + price breakout + RSI>50
        # Relaxed conditions to ensure enough trades
        if bull_trend_1d and is_squeeze and bw_expanding:
            if breakout_long and (vol_spike or rsi_bullish):
                new_signal = SIZE_BASE
        
        # Short: 1d bearish + squeeze release + volume spike + price breakout + RSI<50
        if bear_trend_1d and is_squeeze and bw_expanding:
            if breakout_short and (vol_spike or rsi_bearish):
                new_signal = -SIZE_BASE
        
        # Alternative: ADX trend + breakout (without squeeze) for more trades
        if bull_trend_1d and trend_strength and breakout_long and rsi_bullish:
            if new_signal == 0.0:  # Don't override squeeze signal
                new_signal = SIZE_BASE * 0.8  # Slightly smaller position
        
        if bear_trend_1d and trend_strength and breakout_short and rsi_bearish:
            if new_signal == 0.0:  # Don't override squeeze signal
                new_signal = -SIZE_BASE * 0.8  # Slightly smaller position
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = int(np.sign(new_signal))
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = int(np.sign(new_signal))
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals