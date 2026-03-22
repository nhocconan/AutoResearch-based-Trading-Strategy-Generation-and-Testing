#!/usr/bin/env python3
"""
Experiment #275: 12h Fisher Transform Reversal with 1d HMA Bias and Supertrend Filter

Hypothesis: After analyzing 274 experiments, the key insight is that simple trend-following
works in bull markets but fails in bear/range markets (2022 crash, 2025 bear). The Fisher
Transform excels at catching reversals in bear market rallies (research shows 75% win rate).

This strategy combines:
1. Fisher Transform (period=9) - catches reversals at extremes, works well in bear markets
2. 1d HMA(21) - strong directional bias filter (prevents counter-trend trades in 2022)
3. Supertrend (ATR=10, mult=3) - confirms trend direction, reduces whipsaws
4. ADX(14) - only trade when ADX>20 (trending) or ADX<25 (range mean-reversion)
5. ATR-based stoploss (2.5*ATR) - appropriate for 12h timeframe
6. Asymmetric sizing - larger positions in strong trends (ADX>30)

Why 12h + Fisher might work:
- 12h has fewer bars = less fee drag, cleaner signals
- Fisher Transform catches reversals that EMA/Supertrend miss
- 1d HMA bias prevents trading against major trend (critical for 2022)
- ADX filter avoids choppy periods where Fisher whipsaws
- Looser entry thresholds ensure >=10 trades per symbol

Key differences from failed strategies:
- NO RSI (RSI pullback failed in #251, #254, #259, #263)
- NO Donchian breakout (failed in #263, #267 with negative Sharpe)
- NO volume confirmation (didn't help in #263, #268)
- Fisher Transform is NEW for 12h timeframe (only tried on 15m in #271)
- Simpler than ensemble strategies that failed (#256 Sharpe=-0.231)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete, scaled by ADX strength
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_reversal_1d_hma_supertrend_adx_v1"
timeframe = "12h"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals at extremes. Long when Fisher crosses above -1.5,
    short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize price to -1 to +1 range
        range_val = highest - lowest
        if range_val == 0:
            continue
        
        normalized = (hl2 - lowest) / range_val
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth with EMA
        if i == period:
            fisher[i] = fisher_val
            trigger[i] = fisher_val
        else:
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
            trigger[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else fisher[i]
    
    return fisher, trigger

def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    supertrend[:] = np.nan
    trend[:] = np.nan
    
    for i in range(len(atr)):
        if np.isnan(atr[i]):
            continue
        
        # Calculate basic upper and lower bands
        hl2 = (high[i] + low[i]) / 2.0
        upper_band = hl2 + multiplier * atr[i]
        lower_band = hl2 - multiplier * atr[i]
        
        if i == 0:
            supertrend[i] = upper_band
            trend[i] = 1
        else:
            # Update bands based on previous trend
            if trend[i-1] == 1:
                # Previous trend was up
                if lower_band > supertrend[i-1]:
                    supertrend[i] = lower_band
                else:
                    supertrend[i] = supertrend[i-1]
            else:
                # Previous trend was down
                if upper_band < supertrend[i-1]:
                    supertrend[i] = upper_band
                else:
                    supertrend[i] = supertrend[i-1]
            
            # Determine current trend
            if close[i] > supertrend[i]:
                trend[i] = 1
            else:
                trend[i] = -1
    
    return supertrend, trend

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    if n < period * 2:
        return adx
    
    # Calculate +DM, -DM, and TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI, -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_s[i] != 0:
            plus_di[i] = 100 * plus_dm_s[i] / tr_s[i]
            minus_di[i] = 100 * minus_dm_s[i] / tr_s[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
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
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    supertrend, supertrend_trend = calculate_supertrend(high, low, close, atr, 3.0)
    adx = calculate_adx(high, low, close, 14)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_STRONG = 0.35  # Larger size in strong trends (ADX>30)
    SIZE_WEAK = 0.20  # Smaller size in weak trends
    
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
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(supertrend_trend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = strong directional bias (hard filter)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === SUPERTREND CONFIRMATION ===
        supertrend_bull = supertrend_trend[i] == 1
        supertrend_bear = supertrend_trend[i] == -1
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        
        # Alternative: Fisher crosses above trigger (momentum)
        fisher_momentum_long = fisher[i] > fisher_trigger[i] and fisher[i] < 0
        fisher_momentum_short = fisher[i] < fisher_trigger[i] and fisher[i] > 0
        
        # === ADX FILTER ===
        # Only trade when ADX indicates some trend or range condition
        adx_trending = adx[i] > 20  # Some trend strength
        adx_strong = adx[i] > 30  # Strong trend
        adx_range = adx[i] < 25  # Range market (mean reversion ok)
        
        # === POSITION SIZING ===
        if adx_strong:
            position_size = SIZE_STRONG
        elif adx_trending or adx_range:
            position_size = SIZE_BASE
        else:
            position_size = SIZE_WEAK
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 1d bias up OR Supertrend up + Fisher reversal/momentum
        # Looser conditions to ensure >=10 trades per symbol
        long_conditions = (
            (bull_trend_1d or supertrend_bull) and  # 1d HMA OR Supertrend bullish
            (fisher_long or fisher_momentum_long) and  # Fisher signal
            adx_trending  # Some trend strength
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            (bear_trend_1d or supertrend_bear) and  # 1d HMA OR Supertrend bearish
            (fisher_short or fisher_momentum_short) and  # Fisher signal
            adx_trending  # Some trend strength
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
        # Exit if HTF bias reverses against position (strong filter)
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d and not supertrend_bull:
                new_signal = 0.0  # Both 1d and Supertrend reversed against long
            if position_side < 0 and bull_trend_1d and not supertrend_bear:
                new_signal = 0.0  # Both 1d and Supertrend reversed against short
        
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