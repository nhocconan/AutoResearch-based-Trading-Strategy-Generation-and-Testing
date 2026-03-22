#!/usr/bin/env python3
"""
Experiment #341: 12h Fisher Transform + 1d HMA Trend + ADX Momentum Filter

Hypothesis: After 290+ failed strategies, the key insight is that 12h timeframe needs:
1. Slower HTF filter (1d HMA instead of 4h) for more stable trend bias
2. Fisher Transform for entry signals - proven to catch reversals in bear markets
3. ADX filter to avoid trading during extreme chop (ADX < 20 = no trade)
4. Looser entry thresholds on 12h to ensure >=10 trades per symbol

Why Fisher Transform?
- Normalizes price to Gaussian distribution (-2 to +2 range)
- Crosses at extremes signal reversals better than RSI in bear markets
- Worked well in 2022 crash and 2025 bear market in research papers

Why 1d HMA?
- 12h primary needs slower HTF for stable bias
- HMA has less lag than EMA for trend detection
- 1d boundaries match Binance actual data (no resampling)

Position sizing: 0.25 discrete (conservative for 12h slower signals)
Stoploss: 2.5 * ATR(14) trailing
Entry: Fisher < -1.5 (long) or > +1.5 (short) + ADX > 18 + 1d HMA bias

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_1d_hma_adx_momentum_atr_v1"
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
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution.
    
    Steps:
    1. Calculate typical price: (high + low + close) / 3
    2. Normalize: (price - lowest) / (highest - lowest)
    3. Scale to -0.99 to +0.99
    4. Apply Fisher: 0.5 * ln((1+x)/(1-x))
    
    Entry signals:
    - Fisher crosses above -1.5 from below = long
    - Fisher crosses below +1.5 from above = short
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 5:
        return fisher, fisher_prev
    
    # Typical price
    typical = (high + low + close) / 3.0
    
    for i in range(period, n):
        # Find highest high and lowest low over lookback
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_prev[i] = fisher[i-1] if i > 1 else 0.0
            continue
        
        # Normalize price to 0-1 range
        normalized = (typical[i] - lowest) / price_range
        
        # Scale to -0.99 to +0.99 (avoid division by zero in log)
        scaled = max(-0.99, min(0.99, 2 * normalized - 1))
        
        # Apply Fisher transform
        fisher[i] = 0.5 * np.log((1 + scaled) / (1 - scaled + 1e-10))
        
        # Track previous value for crossover detection
        if i > 0:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    Measures trend strength (not direction).
    ADX > 25 = strong trend, ADX < 20 = weak/ranging
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 5:
        return adx
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth using Wilder's method (EMA with alpha = 1/period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX (smoothed DX)
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx[:] = adx_series.values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25  # Conservative for 12h slower signals
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # Fisher crossover tracking
    prev_fisher_signal = 0  # 0=none, 1=long, -1=short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === 1d HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === ADX MOMENTUM FILTER ===
        # Only trade when ADX > 18 (some momentum, avoid extreme chop)
        has_momentum = adx[i] > 18
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long_cross = (fisher_prev[i] < -1.5 and fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 from above
        fisher_short_cross = (fisher_prev[i] > 1.5 and fisher[i] <= 1.5)
        
        # Also allow entries when Fisher is at extreme (reversal setup)
        fisher_deep_oversold = fisher[i] < -1.8
        fisher_deep_overbought = fisher[i] > 1.8
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Fisher signal + bullish 1d bias + momentum
        if (fisher_long_cross or fisher_deep_oversold) and bull_trend_1d and has_momentum:
            new_signal = SIZE
            prev_fisher_signal = 1
        
        # SHORT ENTRY: Fisher signal + bearish 1d bias + momentum
        elif (fisher_short_cross or fisher_deep_overbought) and bear_trend_1d and has_momentum:
            new_signal = -SIZE
            prev_fisher_signal = -1
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
                    prev_fisher_signal = 0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
                    prev_fisher_signal = 0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1d trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
                prev_fisher_signal = 0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
                prev_fisher_signal = 0
        
        # === ADX DROPS TOO LOW EXIT ===
        # If momentum disappears, exit position
        if in_position and adx[i] < 15:
            new_signal = 0.0
            prev_fisher_signal = 0
        
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