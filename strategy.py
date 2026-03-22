#!/usr/bin/env python3
"""
Experiment #206: 30m Ehlers Fisher Transform + 4h HMA Trend + ADX Regime + Volume + ATR Stop

Hypothesis: The Ehlers Fisher Transform (from "Cybernetic Analysis for Stocks and Futures")
normalizes price into a Gaussian distribution, making extreme values (-1.5 to +1.5) reliable
reversal signals. Combined with 4h HMA trend bias, ADX regime filter, and volume confirmation,
this should catch trend continuations after pullbacks while avoiding choppy whipsaws.

Why 30m + Fisher Transform:
- 30m captures intraday swings without 15m noise
- Fisher Transform proven to identify turning points in trending markets
- 4h HMA provides stable higher-timeframe bias (avoid counter-trend trades)
- ADX > 18 filters choppy periods (learned from failed mean-reversion attempts)
- Volume surge (1.5x avg) confirms breakout validity
- Conservative sizing (0.25) with 2.5*ATR stop controls drawdown

Learning from failures:
- #194, #199, #200, #201, #205: Mean reversion fails on crypto (negative Sharpe)
- #195, #196, #202: Need simpler regime logic, not over-engineered
- #197, #198, #203, #204: Trend-following with HTF filter works better
- Current best #4h_kama_1d_hma: Sharpe=0.478, need to beat this

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_4h_hma_adx_vol_atr_v1"
timeframe = "30m"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
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
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price into Gaussian distribution for clearer reversal signals.
    Reference: "Cybernetic Analysis for Stocks and Futures" by John Ehlers
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Calculate median price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize to -1 to +1 range
        range_val = highest - lowest
        if range_val < 1e-10:
            range_val = 1e-10
        
        x = (hl2 - lowest) / range_val
        
        # Clamp to avoid division issues
        x = np.clip(x, 0.001, 0.999)
        
        # Fisher transform formula
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Trigger line (previous fisher value)
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    # Fill initial values
    fisher[:period] = fisher[period] if period < n else 0.0
    trigger[:period] = trigger[period] if period < n else 0.0
    
    return fisher, trigger

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    fisher, trigger = calculate_fisher_transform(high, low, 9)
    vol_sma = calculate_volume_sma(volume, 20)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(fisher[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        if vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 18 = trending market (avoid choppy whipsaws)
        trend_strength = adx[i] > 18
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crossing above -1.5 from below = bullish reversal
        # Fisher crossing below +1.5 from above = bearish reversal
        fisher_bullish = (fisher[i] > -1.5) and (trigger[i] <= -1.5)
        fisher_bearish = (fisher[i] < 1.5) and (trigger[i] >= 1.5)
        
        # Also check extreme oversold/overbought for continuation
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === VOLUME CONFIRMATION ===
        # Volume surge = 1.5x average confirms breakout validity
        volume_surge = volume[i] > 1.5 * vol_sma[i]
        
        # === EMA CONFIRMATION ===
        # EMA21 > EMA50 = bullish trend structure
        # EMA21 < EMA50 = bearish trend structure
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + ADX trending + Fisher bullish reversal + (volume OR EMA bullish)
        # More flexible to ensure enough trades
        if bull_trend_4h and trend_strength:
            # Fisher reversal OR Fisher oversold with EMA support
            if fisher_bullish or (fisher_oversold and ema_bullish):
                # Need volume confirmation OR strong EMA structure
                if volume_surge or (ema_bullish and close[i] > ema_21[i]):
                    new_signal = SIZE_BASE
        
        # Short: 4h bearish + ADX trending + Fisher bearish reversal + (volume OR EMA bearish)
        if bear_trend_4h and trend_strength:
            # Fisher reversal OR Fisher overbought with EMA resistance
            if fisher_bearish or (fisher_overbought and ema_bearish):
                # Need volume confirmation OR strong EMA structure
                if volume_surge or (ema_bearish and close[i] < ema_21[i]):
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
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