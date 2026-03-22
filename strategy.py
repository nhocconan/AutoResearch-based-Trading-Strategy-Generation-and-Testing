#!/usr/bin/env python3
"""
Experiment #274: 4h KAMA Trend with 1d HMA Bias and ADX Regime Filter

Hypothesis: After analyzing 273 experiments, the pattern is clear:
- Complex ensembles with RSI fail consistently (#251, #254, #259, #262)
- Simple trend-following with strong HTF bias works best
- Current best (mtf_4h_kama_1d_hma_adx_atr_v1) has Sharpe=0.478

This strategy improves by:
1. Using KAMA(21) on 4h - adaptive to volatility, less lag than EMA
2. 1d HMA(21) for directional bias - prevents counter-trend trades in 2022 crash
3. ADX(14) regime filter - only trade when ADX>20 (trending, not choppy)
4. Asymmetric entry - only long when 1d HMA bullish, only short when bearish
5. 2.5*ATR trailing stoploss - appropriate for 4h timeframe
6. Discrete position sizing (0.25/0.30) to minimize fee churn
7. Looser ADX threshold (20 not 25) to ensure >=10 trades per symbol

Why this might beat the current best:
- KAMA adapts to volatility better than HMA in ranging markets
- ADX filter removes whipsaw trades during low-volatility periods
- Asymmetric entries prevent fighting the HTF trend
- Simpler logic = fewer conflicting conditions = more trades

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_trend_1d_hma_adx_asymmetric_v1"
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

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - moves fast in trends, slow in chop.
    Formula: KAMA = KAMA_prev + SC * (Price - KAMA_prev)
    SC = (ER * (fast_SC - slow_SC) + slow_SC)^2
    ER = Change / Sum of absolute changes over efficiency_period
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, efficiency_period))
    change[:efficiency_period] = np.nan
    
    sum_change = np.zeros(n)
    sum_change[:] = np.nan
    for i in range(efficiency_period, n):
        sum_change[i] = np.sum(np.abs(close[i-efficiency_period+1:i+1] - np.roll(close[i-efficiency_period+1:i+1], 1)))
    
    er = change / sum_change
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[efficiency_period] = close[efficiency_period]
    
    # Calculate KAMA
    for i in range(efficiency_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
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
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending, ADX < 20 = ranging/choppy.
    """
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
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
    
    # Smooth TR, +DM, -DM using Wilder's method (EMA with span=period)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_s > 0
    plus_di[mask] = 100.0 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100.0 * minus_dm_s[mask] / tr_s[mask]
    
    # Calculate DX and ADX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    
    dx = np.zeros(n)
    mask2 = di_sum > 0
    dx[mask2] = 100.0 * di_diff[mask2] / di_sum[mask2]
    
    # ADX is EMA of DX
    adx_s = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_s.values
    
    return adx

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama_4h = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    adx_4h = calculate_adx(high, low, close, 14)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size (conservative)
    SIZE_STRONG = 0.30  # Stronger signal size
    SIZE_MAX = 0.35  # Maximum position size
    
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
        
        if np.isnan(kama_4h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_4h[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = strong directional bias (hard filter)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND REGIME FILTER ===
        # ADX > 20 = trending market (trade with trend)
        # ADX < 20 = choppy market (stay flat or reduce size)
        trending_market = adx_4h[i] > 20.0
        
        # === 4H TREND SIGNAL ===
        # KAMA crossover with price
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # Additional confirmation: price above/below 50 EMA
        ema_bullish = close[i] > ema_50[i] if not np.isnan(ema_50[i]) else False
        ema_bearish = close[i] < ema_50[i] if not np.isnan(ema_50[i]) else False
        
        # === POSITION SIZING ===
        # Increase size when ADX is strong (>30 = very trending)
        if adx_4h[i] > 30.0:
            position_size = SIZE_STRONG
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 1d bias up + 4h KAMA bullish + ADX trending + EMA confirmation
        # Looser conditions to ensure >=10 trades per symbol
        long_conditions = (
            bull_trend_1d and  # 1d HMA bias bullish
            kama_bullish and  # 4h KAMA bullish
            trending_market and  # ADX confirms trending
            ema_bullish  # Price above 50 EMA
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_1d and  # 1d HMA bias bearish
            kama_bearish and  # 4h KAMA bearish
            trending_market and  # ADX confirms trending
            ema_bearish  # Price below 50 EMA
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
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0  # 1d trend reversed against long
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0  # 1d trend reversed against short
        
        # === ADX DROPOUT EXIT ===
        # Exit if market becomes choppy (ADX falls below 18)
        if in_position and adx_4h[i] < 18.0:
            new_signal = 0.0  # Market no longer trending
        
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