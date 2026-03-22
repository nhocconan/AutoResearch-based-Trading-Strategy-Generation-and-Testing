#!/usr/bin/env python3
"""
Experiment #273: 1h KAMA Trend with 4h HMA Bias and 12h ADX Regime Filter

Hypothesis: The 1h timeframe sits in a difficult zone - too noisy for pure trend,
too slow for mean reversion. Success requires strong HTF filtering:

1. 4h HMA(21) for directional bias - proven in best strategy (Sharpe=0.478)
2. 1h KAMA(14) for adaptive trend following - adjusts to volatility
3. 12h ADX(14) for regime filter - only trade when ADX > 20 (trending regime)
4. ATR(14) trailing stoploss at 2.5*ATR - tighter than 12h strategies
5. Asymmetric entries - only long when 4h HMA bullish, only short when bearish
6. Volume confirmation - but loose threshold (1.2x not 1.5x) to ensure trades

Why this might beat the 4h baseline:
- 1h has 4x more bars = more entry opportunities
- KAMA adapts faster than EMA in volatile regimes
- 12h ADX filter prevents choppy market whipsaws
- Still uses proven 4h HMA bias from best strategy

Key differences from failed 1h strategies:
- NO RSI (failed in #261, #267)
- NO complex ensemble voting (failed in #256)
- NO mean reversion (failed in #261 with Sharpe=-24)
- Simple: HTF bias + adaptive trend + regime filter

Position sizing: 0.25 base, 0.35 max (discrete levels)
Stoploss: 2.5 * ATR(14) trailing
Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h HMA + 12h ADX via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_trend_4h_hma_12h_adx_atr_v1"
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

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman's Adaptive Moving Average.
    KAMA adapts to market noise - smooth in ranging, responsive in trending.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constant
    sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        # True Range
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        plus_dm[i] = high[i] - high[i-1] if (high[i] - high[i-1] > low[i-1] - low[i] and high[i] - high[i-1] > 0) else 0
        minus_dm[i] = low[i-1] - low[i] if (low[i-1] - low[i] > high[i] - high[i-1] and low[i-1] - low[i] > 0) else 0
    
    # Smooth DM and TR
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    # Use EMA for smoothing (faster than Wilder's)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 0:
            plus_di[i] = 100 * plus_dm_s[i] / tr_s[i]
            minus_di[i] = 100 * minus_dm_s[i] / tr_s[i]
        else:
            plus_di[i] = 0
            minus_di[i] = 0
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # ADX = EMA of DX
    dx_s = pd.Series(dx)
    adx_vals = dx_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:] = adx_vals
    
    return adx

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
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
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 14, 2, 30)
    ema_50 = calculate_ema(close, 50)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_MAX = 0.35  # Maximum position size
    SIZE_MIN = 0.15  # Minimum position size
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS (4h HMA) ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (12h ADX) ===
        # Only trade when ADX > 20 (trending regime, not choppy)
        trending_regime = adx_12h_aligned[i] > 20.0
        
        # === VOLUME CONFIRMATION ===
        # Loose threshold to ensure >=10 trades per symbol
        volume_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # === KAMA TREND SIGNAL ===
        # KAMA crossover with EMA50 for entry timing
        kama_above_ema = kama[i] > ema_50[i]
        kama_below_ema = kama[i] < ema_50[i]
        
        # KAMA slope (momentum confirmation)
        kama_slope_up = kama[i] > kama[i-1] if i > 0 else False
        kama_slope_down = kama[i] < kama[i-1] if i > 0 else False
        
        # === POSITION SIZING ===
        # Increase size when trend is strong (ADX > 30)
        strong_trend = adx_12h_aligned[i] > 30.0
        if strong_trend:
            position_size = SIZE_MAX
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 4h bias up + trending regime + KAMA signal + volume
        # Looser conditions to ensure >=10 trades per symbol
        long_conditions = (
            bull_trend_4h and  # 4h HMA bias bullish
            trending_regime and  # 12h ADX confirms trending
            kama_above_ema and  # KAMA above EMA50
            kama_slope_up and  # KAMA sloping up
            volume_confirmed  # Volume confirms
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_4h and  # 4h HMA bias bearish
            trending_regime and  # 12h ADX confirms trending
            kama_below_ema and  # KAMA below EMA50
            kama_slope_down and  # KAMA sloping down
            volume_confirmed  # Volume confirms
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
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
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
        # === REGIME EXIT ===
        # Exit if regime becomes choppy (ADX < 18 with hysteresis)
        if in_position and adx_12h_aligned[i] < 18.0:
            new_signal = 0.0  # Regime no longer trending
        
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
            # else: maintaining same position direction (possibly adjusted size)
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