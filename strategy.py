#!/usr/bin/env python3
"""
Experiment #223: 15m Mean Reversion + 4h HMA Trend + 1h RSI + Volume Filter

Hypothesis: 15m timeframe is too noisy for pure trend-following (see #211, #217 failures).
Instead, use mean reversion on 15m WITH strong HTF trend filter. Enter on RSI extremes
(oversold in uptrend, overbought in downtrend) when 4h HMA confirms direction.
Volume spike confirms genuine moves vs noise. This should work better than pure
MACD/Supertrend approaches that failed on 15m/30m.

Why 15m mean reversion + HTF trend might work:
- 15m has more noise, but mean reversion exploits this (buy dips in uptrend)
- 4h HMA provides stable trend bias (avoid counter-trend mean reversion)
- 1h RSI aligned gives intermediate momentum confirmation
- Volume filter ensures we're not catching dead-cat bounces
- Conservative sizing (0.25) controls drawdown

Learning from failures:
- #211 (15m MACD): Sharpe=-2.49 - pure momentum fails on noisy TF
- #217 (15m KAMA): Sharpe=-1.93 - trend following whipsaws on 15m
- #212 (30m Chop): Sharpe=-2.70 - regime detection alone insufficient
- Mean reversion WITH trend filter works better than pure trend on lower TF

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h HMA trend + 1h RSI momentum (both via mtf_data helper)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_meanrev_4h_hma_1h_rsi_vol_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_bb(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper.values, lower.values, sma.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume / (vol_avg.values + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bb(close, 20, 2.0)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_15m[i]) or np.isnan(bb_upper[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1H RSI MOMENTUM ===
        # RSI > 50 = bullish momentum on 1h
        # RSI < 50 = bearish momentum on 1h
        rsi_1h_bullish = rsi_1h_aligned[i] > 50
        rsi_1h_bearish = rsi_1h_aligned[i] < 50
        
        # === 15M RSI MEAN REVERSION ===
        # RSI < 35 = oversold (potential long in uptrend)
        # RSI > 65 = overbought (potential short in downtrend)
        rsi_oversold = rsi_15m[i] < 35
        rsi_overbought = rsi_15m[i] > 65
        
        # === BOLLINGER BAND POSITION ===
        # Price near lower band in uptrend = good long entry
        # Price near upper band in downtrend = good short entry
        price_near_lower = close[i] < bb_lower[i] * 1.02  # Within 2% of lower
        price_near_upper = close[i] > bb_upper[i] * 0.98  # Within 2% of upper
        
        # === VOLUME CONFIRMATION ===
        # Volume ratio > 1.2 = above average volume (confirms move)
        vol_confirmed = vol_ratio[i] > 1.2
        
        # === EMA TREND STRUCTURE ===
        # EMA21 > EMA50 = bullish structure
        # EMA21 < EMA50 = bearish structure
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + 1h RSI bullish + 15m RSI oversold + (BB lower OR volume)
        # Flexible conditions to ensure enough trades
        if bull_trend_4h and rsi_1h_bullish:
            if rsi_oversold and (price_near_lower or vol_confirmed):
                new_signal = SIZE_BASE
            elif rsi_15m[i] < 40 and ema_bullish:
                # Alternative: less oversold but EMA confirms
                new_signal = SIZE_BASE
        
        # Short: 4h bearish + 1h RSI bearish + 15m RSI overbought + (BB upper OR volume)
        if bear_trend_4h and rsi_1h_bearish:
            if rsi_overbought and (price_near_upper or vol_confirmed):
                new_signal = -SIZE_BASE
            elif rsi_15m[i] > 60 and ema_bearish:
                # Alternative: less overbought but EMA confirms
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
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