#!/usr/bin/env python3
"""
Experiment #289: 15m Supertrend with 4h HMA Bias and ADX Regime Filter

Hypothesis: After analyzing 288 failed experiments, the pattern shows that:
1. Complex ensembles with too many filters = 0 trades (Sharpe=0.000)
2. RSI pullback strategies consistently failed across all timeframes
3. Simple trend + strong HTF bias + regime filter = best combination
4. 15m timeframe needs wider stops and looser entries than 4h/12h

This strategy uses:
1. 4h HMA(21) for directional bias - prevents counter-trend trades (proven edge)
2. 15m Supertrend(10,3) for entry timing - catches momentum breaks
3. ADX(14) > 20 regime filter - only trade in trending markets
4. ATR-based stoploss at 2.5*ATR - appropriate for 15m noise
5. Discrete position sizing: 0.0, ±0.25, ±0.30
6. LOOSE entry conditions to ensure >=10 trades per symbol

Why this might work on 15m:
- Supertrend captures momentum better than EMA crossover on intraday
- 4h HMA bias is strong enough to filter 2022 crash direction
- ADX > 20 is achievable (unlike ADX > 40 which rarely triggers)
- Fewer false signals than 5m, more opportunities than 1h

Key differences from failed 15m strategies (#277, #283):
- NO RSI (RSI pullback failed in #277, #284, #285)
- NO complex voting systems
- Simpler entry logic = more trades
- Looser ADX threshold (20 vs 25+)
- Position size 0.25-0.30 (not 0.35 which caused high DD)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.30 max, discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_adx_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    # Upper and lower bands
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if direction[i-1] == 1:
            # Previously long
            if close[i] > supertrend[i-1]:
                # Stay long, update lower band
                supertrend[i] = max(supertrend[i-1], lower_band[i])
                direction[i] = 1
            else:
                # Flip to short
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previously short
            if close[i] < supertrend[i-1]:
                # Stay short, update upper band
                supertrend[i] = min(supertrend[i-1], upper_band[i])
                direction[i] = -1
            else:
                # Flip to long
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    # Calculate DM and TR
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
    
    # Calculate DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_s > 0
    plus_di[mask] = 100 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100 * minus_dm_s[mask] / tr_s[mask]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    mask2 = (plus_di + minus_di) > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, atr, multiplier=3.0)
    adx = calculate_adx(high, low, close, 14)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_MAX = 0.30  # Maximum position size
    
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
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = strong directional bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER ===
        # ADX > 20 = trending market (loose threshold to ensure trades)
        trending_regime = adx[i] > 20
        
        # === SUPERTREND SIGNAL ===
        # st_direction: 1 = long signal, -1 = short signal
        supertrend_long = st_direction[i] == 1
        supertrend_short = st_direction[i] == -1
        
        # === ENTRY CONDITIONS (LOOSE to ensure >=10 trades) ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h bias up + Supertrend long + ADX trending
        # Only need 2 of 3 conditions for long (looser than short)
        long_score = int(bull_trend_4h) + int(supertrend_long) + int(trending_regime)
        
        # SHORT ENTRY: 4h bias down + Supertrend short + ADX trending
        short_score = int(bear_trend_4h) + int(supertrend_short) + int(trending_regime)
        
        # Generate signal based on score (need at least 2 conditions)
        if long_score >= 2 and supertrend_long:
            new_signal = SIZE_BASE
        
        if short_score >= 2 and supertrend_short:
            new_signal = -SIZE_BASE
        
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
        # Exit if 4h HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
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