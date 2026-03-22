#!/usr/bin/env python3
"""
Experiment #293: 12h EMA Crossover with 1d HMA Bias and Fisher Transform Entry

Hypothesis: After analyzing 292 experiments, clear patterns emerge:
1. 4h timeframe with 1d HMA bias works best (current Sharpe=0.485)
2. 12h has potential (#287 Sharpe=0.370) but needs simpler logic
3. RSI pullbacks consistently FAIL across all timeframes
4. Complex ensembles always underperform simple trend following
5. ADX filters help but shouldn't be too restrictive

This strategy combines:
1. 1d HMA(21) for strong directional bias (proven edge from #292)
2. 12h EMA(13)/EMA(50) crossover for entry timing (simpler than Donchian)
3. Ehlers Fisher Transform(9) for entry confirmation (catches reversals better than RSI)
4. ADX(14)>18 for trend strength (looser than typical 25 threshold)
5. ATR-based position sizing and 3.5*ATR trailing stoploss

Why this might beat #292:
- 12h has fewer false signals than 4h (less fee drag)
- Fisher Transform catches trend reversals earlier than EMA alone
- Looser ADX threshold (18 vs 25) ensures >=10 trades per symbol
- Simpler than Donchian breakout (#263) while maintaining trend bias

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 3.5 * ATR(14) trailing (wider for 12h timeframe)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_ema_fisher_1d_hma_adx_atr_v1"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - transforms price into Gaussian distribution
    for clearer reversal signals. Period=9 is standard.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: 0.66 * ((price - lowest) / (highest - lowest) - 0.5)
    3. Smooth with EMA
    4. Fisher = 0.5 * ln((1 + value) / (1 - value))
    5. Signal line = previous Fisher value
    """
    n = len(high)
    typical = (high + low) / 2.0
    
    # Normalize price over lookback period
    normalized = np.zeros(n)
    normalized[:] = np.nan
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        if highest > lowest:
            normalized[i] = 0.66 * ((typical[i] - lowest) / (highest - lowest) - 0.5)
        else:
            normalized[i] = 0.0
    
    # Clip to avoid ln domain errors
    normalized = np.clip(normalized, -0.99, 0.99)
    
    # Smooth with EMA
    norm_s = pd.Series(normalized)
    smoothed = norm_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Fisher transform
    fisher = np.zeros(n)
    fisher[:] = np.nan
    for i in range(period, n):
        if np.abs(smoothed[i]) < 0.99:
            fisher[i] = 0.5 * np.log((1 + smoothed[i]) / (1 - smoothed[i]))
    
    # Signal line (previous Fisher value)
    signal = np.roll(fisher, 1)
    signal[0] = np.nan
    
    return fisher, signal

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending, ADX < 20 = ranging.
    We use 18 as threshold for 12h timeframe.
    """
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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
    ema_fast = calculate_ema(close, 13)
    ema_slow = calculate_ema(close, 50)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30  # Base position size
    SIZE_REDUCED = 0.20  # Reduced size in high vol
    SIZE_INCREASED = 0.35  # Increased size in strong trend
    
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
        
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = strong directional bias (hard filter from #292)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH ===
        # ADX > 18 = trending market (looser than 25 for 12h)
        trending = adx[i] > 18
        
        # === EMA CROSSOVER ===
        # Fast EMA above slow EMA = bullish momentum
        ema_bullish = ema_fast[i] > ema_slow[i]
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crossing above signal line = bullish reversal
        fisher_bullish = fisher[i] > fisher_signal[i] and fisher_signal[i] < -0.5
        # Fisher crossing below signal line = bearish reversal
        fisher_bearish = fisher[i] < fisher_signal[i] and fisher_signal[i] > 0.5
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when ATR is elevated (>1.5x recent average)
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Strong trend = increase size
        strong_trend = adx[i] > 30
        
        # Determine position size based on volatility and trend strength
        if high_volatility:
            position_size = SIZE_REDUCED
        elif strong_trend:
            position_size = SIZE_INCREASED
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 1d bias up + EMA bullish + Fisher confirmation
        # Looser conditions to ensure >=10 trades per symbol
        long_conditions = (
            bull_trend_1d and  # 1d HMA bias bullish
            ema_bullish and  # EMA crossover bullish
            (fisher_bullish or trending)  # Fisher reversal OR strong trend
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_1d and  # 1d HMA bias bearish
            ema_bearish and  # EMA crossover bearish
            (fisher_bearish or trending)  # Fisher reversal OR strong trend
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 3.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 3.5 * ATR below highest close
                stoploss_price = highest_close - 3.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 3.5 * ATR above lowest close
                stoploss_price = lowest_close + 3.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0  # 1d trend reversed against long
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0  # 1d trend reversed against short
        
        # === EMA CROSSOVER EXIT ===
        # Exit if EMA crossover reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and ema_bearish:
                new_signal = 0.0  # EMA crossed against long
            if position_side < 0 and ema_bullish:
                new_signal = 0.0  # EMA crossed against short
        
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