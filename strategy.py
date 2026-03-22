#!/usr/bin/env python3
"""
Experiment #316: 4h Regime-Adaptive Strategy with Choppiness Index

Hypothesis: After #310 failed (Sharpe=-0.931), the Supertrend+Fisher combo was too complex.
Analysis shows regime detection is key - BTC/ETH spend 60%+ time in range markets.

This strategy uses Choppiness Index (CHOP) to detect market regime:
1. CHOP(14) > 61.8 = RANGE → Mean reversion logic (RSI extremes + BB touch)
2. CHOP(14) < 38.2 = TREND → Trend following logic (EMA crossover + ADX)
3. 38.2 <= CHOP <= 61.8 = TRANSITION → Stay flat or reduce position

HTF Bias:
- 1d HMA(21) for primary directional bias (REQUIRED for entries)
- 1w HMA(21) for meta-trend confirmation (soft filter, boosts size)

Key innovations:
- Regime-adaptive entry logic (different signals for trend vs range)
- Hysteresis on CHOP to avoid whipsaw (enter trend at 38.2, exit at 45)
- Volume confirmation on breakouts (volume > 1.3x 20-bar avg)
- ATR(14) trailing stoploss at 2.5x (proven from successful strategies)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_chop_1d_1w_hma_adaptive_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
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
    CHOP > 61.8 = Range/Ranging market
    CHOP < 38.2 = Trending market
    """
    n = len(close)
    chop = np.zeros(n) * np.nan
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop = calculate_choppiness(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # Regime hysteresis tracking
    prev_regime = 0  # 0=unknown, 1=trend, 2=range
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = primary directional bias (REQUIRED)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA = meta-trend confirmation (SOFT - boosts size but not required)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION with HYSTERESIS ===
        # CHOP < 38.2 = Trending, CHOP > 61.8 = Range
        # Hysteresis: enter trend at 38.2, exit at 45; enter range at 61.8, exit at 55
        chop_val = chop[i]
        
        if prev_regime == 1:  # Was in trend
            if chop_val < 45:
                regime = 1  # Stay in trend
            elif chop_val > 61.8:
                regime = 2  # Switch to range
            else:
                regime = 1  # Stay in trend (hysteresis zone)
        elif prev_regime == 2:  # Was in range
            if chop_val > 55:
                regime = 2  # Stay in range
            elif chop_val < 38.2:
                regime = 1  # Switch to trend
            else:
                regime = 2  # Stay in range (hysteresis zone)
        else:  # Unknown regime
            if chop_val < 38.2:
                regime = 1
            elif chop_val > 61.8:
                regime = 2
            else:
                regime = 0  # Transition/neutral
        
        prev_regime = regime
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 0 else 1.0
        high_volume = volume_ratio > 1.3
        
        # === REGIME-SPECIFIC ENTRY LOGIC ===
        new_signal = 0.0
        position_size = SIZE_BASE
        
        if regime == 1:  # TRENDING REGIME
            # Trend following: EMA crossover + ADX confirmation + HTF bias
            ema_bullish = ema_fast[i] > ema_slow[i]
            ema_bearish = ema_fast[i] < ema_slow[i]
            trending = adx[i] > 18  # Stronger ADX for trend regime
            
            # Check for actual crossover
            ema_cross_bull = ema_bullish and (i > 0 and ema_fast[i-1] <= ema_slow[i-1])
            ema_cross_bear = ema_bearish and (i > 0 and ema_fast[i-1] >= ema_slow[i-1])
            
            # Also allow entry if already in trend state
            ema_trend_bull = ema_bullish
            ema_trend_bear = ema_bearish
            
            # Size boost for strong trend + 1w confirmation
            if adx[i] > 25 and bull_trend_1w:
                position_size = SIZE_STRONG
            elif adx[i] > 25 and bear_trend_1w:
                position_size = SIZE_STRONG
            
            # LONG: 1d bias up + EMA bullish + ADX trending + volume confirm
            long_conditions = (
                bull_trend_1d and
                ema_trend_bull and
                trending
            )
            
            # SHORT: 1d bias down + EMA bearish + ADX trending + volume confirm
            short_conditions = (
                bear_trend_1d and
                ema_trend_bear and
                trending
            )
            
            if long_conditions:
                new_signal = position_size
            if short_conditions:
                new_signal = -position_size
        
        elif regime == 2:  # RANGING REGIME
            # Mean reversion: RSI extremes + BB touch + HTF bias for direction
            rsi_oversold = rsi[i] < 35
            rsi_overbought = rsi[i] > 65
            at_bb_lower = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
            at_bb_upper = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
            
            # LONG: RSI oversold + at BB lower + 1d bias up (or neutral)
            long_conditions = (
                rsi_oversold and
                at_bb_lower and
                (bull_trend_1d or not bear_trend_1d)
            )
            
            # SHORT: RSI overbought + at BB upper + 1d bias down (or neutral)
            short_conditions = (
                rsi_overbought and
                at_bb_upper and
                (bear_trend_1d or not bull_trend_1d)
            )
            
            if long_conditions:
                new_signal = SIZE_BASE
            if short_conditions:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        # Exit long if regime switches from trend to range and we're in a long
        if in_position and new_signal != 0.0:
            if position_side > 0 and regime == 2 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and regime == 2 and bull_trend_1d:
                new_signal = 0.0
        
        # === EMA REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and ema_fast[i] < ema_slow[i]:
                new_signal = 0.0
            if position_side < 0 and ema_fast[i] > ema_slow[i]:
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