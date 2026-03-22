#!/usr/bin/env python3
"""
Experiment #087: 1h Regime-Adaptive Strategy with Choppiness Index + 4h HMA Trend Filter
Hypothesis: 1h timeframe failed before (#075, #081) because they used single-regime logic.
This strategy adapts to market regime using Choppiness Index (CHOP):
- CHOP > 61.8 = Range/Chop → Mean reversion (RSI extremes, Bollinger bounds)
- CHOP < 38.2 = Trend → Trend following (Supertrend, EMA alignment)
- 38.2-61.8 = Transition → Reduce position size or stay flat

HTF: 4h HMA for trend bias (faster than 1d, appropriate for 1h entries).
Volume confirmation on breakouts to avoid false signals.
Conservative sizing (0.20-0.30) with 2.5*ATR trailing stops.

Why this might work on 1h (learning from failures):
- #075 (1h Z-score): -2.734 Sharpe - pure mean reversion, no regime filter
- #081 (1h Supertrend): -2.382 Sharpe - pure trend following, whipsaw in ranges
- #076 (4h Supertrend): +0.162 Sharpe - 4h is better TF, but we must use 1h here

Key innovation: Regime-adaptive logic switches between mean reversion and trend following
based on Choppiness Index. This handles both 2021-2024 (trending) and 2025 (range/bear).

Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
Position sizing: 0.20 base, 0.30 strong signals, discrete levels.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_chop_4h_hma_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, trend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = long, -1 = short
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = 1
        else:
            if trend[i-1] == 1:
                if close[i] < lower_band[i-1]:
                    trend[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    trend[i] = 1
                    supertrend[i] = max(upper_band[i], supertrend[i-1])
            else:
                if close[i] > upper_band[i-1]:
                    trend[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    trend[i] = -1
                    supertrend[i] = min(lower_band[i], supertrend[i-1])
    
    return supertrend, trend

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = Range/Chop
    CHOP < 38.2 = Trend
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    atr = calculate_atr(high, low, close, 14)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(atr[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Supertrend
    supertrend, supertrend_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Choppiness Index (regime filter)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    # Volume MA
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Donchian Channel
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
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
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range_regime = chop[i] > 61.8
        is_trend_regime = chop[i] < 38.2
        # 38.2-61.8 = transition (reduce size or flat)
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.2 * vol_ma[i] if vol_ma[i] > 0 else False
        
        # === SUPERTREND SIGNAL ===
        supertrend_long = supertrend_direction[i] == 1
        supertrend_short = supertrend_direction[i] == -1
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === BOLLINGER POSITION ===
        price_near_bb_lower = close[i] < bb_lower[i] * 1.005  # Within 0.5% of lower band
        price_near_bb_upper = close[i] > bb_upper[i] * 0.995  # Within 0.5% of upper band
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        new_signal = 0.0
        
        # === TREND REGIME (CHOP < 38.2) - Trend Following ===
        if is_trend_regime:
            # LONG: Supertrend + 4h bullish + EMA bullish + volume confirmation
            if supertrend_long and bull_trend_4h and ema_bullish:
                if volume_confirmed:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
            
            # SHORT: Supertrend + 4h bearish + EMA bearish + volume confirmation
            if new_signal == 0.0 and supertrend_short and bear_trend_4h and ema_bearish:
                if volume_confirmed:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
        
        # === RANGE REGIME (CHOP > 61.8) - Mean Reversion ===
        elif is_range_regime:
            # LONG: RSI oversold + price near BB lower + 4h not strongly bearish
            if rsi_oversold and price_near_bb_lower and not bear_trend_4h:
                new_signal = SIZE_BASE
            
            # SHORT: RSI overbought + price near BB upper + 4h not strongly bullish
            if new_signal == 0.0 and rsi_overbought and price_near_bb_upper and not bull_trend_4h:
                new_signal = -SIZE_BASE
        
        # === TRANSITION REGIME (38.2 <= CHOP <= 61.8) - Reduced Activity ===
        else:
            # Only take strong signals in transition
            # LONG: All trend conditions + volume
            if supertrend_long and bull_trend_4h and ema_bullish and volume_confirmed:
                new_signal = SIZE_BASE
            
            # SHORT: All trend conditions + volume
            if new_signal == 0.0 and supertrend_short and bear_trend_4h and ema_bearish and volume_confirmed:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5*ATR trailing stop ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals